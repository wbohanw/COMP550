import argparse
import os
import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model_new import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation_revised import to_official, official_evaluate
import wandb
from losses import get_label, get_at_loss, get_balance_loss, get_af_loss, get_sat_loss, get_mean_sat_loss, \
    get_relu_sat_loss, get_margin_loss, compute_contrastive_loss, get_improved_aml_loss
from tqdm import tqdm


def train(args, model, train_features, dev_features, test_features):
    """Train the model and evaluate on the dev set."""

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        best_dev_output = None
        best_model_state = None 
        result = {"dev": {}, "test": {}}  

        train_dataloader = DataLoader(
            features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True
        )
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        lambda_cl = args.lambda_cl
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(int(num_epoch)):
            print(f"Epoch {epoch + 1}/{int(num_epoch)}:")
            model.zero_grad()

            with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
                    wandb.log({"epoch": epoch + 1, "step": num_steps, "lr": cur_lr}, step=num_steps)
                    model.train()
                    inputs = {
                        'input_ids': batch[0].to('cuda'),
                        'attention_mask': batch[1].to('cuda'),
                        'entity_pos': batch[3],
                        'hts': batch[4],
                        'labels': batch[2].to('cuda'),
                    }
                    labels = batch[2].to('cuda')

                    with torch.amp.autocast('cuda'):
                        if args.use_cl:
                            logits, contrastive_features = model(**inputs, return_contrastive_features=True)
                            cl_loss = compute_contrastive_loss(
                                hs_pos=contrastive_features["positive"]["head"],
                                ts_pos=contrastive_features["positive"]["tail"],
                                rs_pos=contrastive_features["positive"]["relation"],
                                hs_neg=contrastive_features["negative"]["head"],
                                ts_neg=contrastive_features["negative"]["tail"],
                                rs_neg=contrastive_features["negative"]["relation"],
                                temperature=args.cl_temperature
                            )
                        else:
                            logits = model(**inputs)
                            cl_loss = 0.0

                        loss_fn_mapping = {
                            'ATL': get_at_loss,
                            'balance_softmax': get_balance_loss,
                            'AFL': get_af_loss,
                            'SAT': get_sat_loss,
                            'MeanSAT': get_mean_sat_loss,
                            'HingeABL': lambda l, lbl: get_relu_sat_loss(l, lbl, args.margin),
                            'AML': get_margin_loss,
                            'New_AML': lambda l, lbl: get_improved_aml_loss(
                                l, lbl,
                                initial_margin=args.margin,
                                num_hard_negatives=5,
                                alpha=0.5,
                                beta=0.5,
                                smoothing=0.1,
                                current_step=num_steps,
                                decay_rate=0.01
                            ),
                        }
                        task_loss_fn = loss_fn_mapping.get(args.loss_type, lambda *_: 0.0)
                        task_loss = task_loss_fn(logits, labels)

                        if args.adjust_lambda_cl and args.use_cl:
                            lambda_cl = min(args.lambda_cl_max, lambda_cl + args.lambda_cl_step)
                            wandb.log({"lambda_cl": lambda_cl}, step=num_steps)

                        loss = lambda_cl * cl_loss + (1 - lambda_cl) * task_loss if args.use_cl else task_loss

                    wandb.log({
                        "cl_loss": cl_loss.item() if isinstance(cl_loss, torch.Tensor) else 0.0,
                        "task_loss": task_loss.item() if isinstance(task_loss, torch.Tensor) else 0.0,
                        "total_loss": loss.item(),
                    }, step=num_steps)

                    loss = loss / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        if args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        model.zero_grad()
                        num_steps += 1

                    pbar.update(1)

                # dev
                dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                print(f"Epoch {epoch + 1}: Dev F1 = {dev_output}")
                wandb.log(dev_output, step=num_steps)

                if dev_score > best_score:
                    best_score = dev_score
                    best_dev_output = dev_output
                    best_model_state = model.state_dict()  

        # test
        model.load_state_dict(best_model_state)  
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)
        wandb.log(test_output)

        # result 
        result["dev"] = best_dev_output
        result["test"] = test_output
        result_path = os.path.join("./result", f"{args.save_name}_result.json")
        os.makedirs("result", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {result_path}")

        # 
        save_path = os.path.join(f"{args.save_path}", f"{args.save_name}.bin")
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")

        return best_dev_output

    new_layer = ["extractor", "bilinear", "projector", "classifier"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    num_steps = 0
    set_seed(args)
    model.zero_grad()
    best_dev_output = finetune(train_features, optimizer, args.num_train_epochs, num_steps)

    return best_dev_output



def evaluate(args, model, features, tag="dev"):
    """Evaluate the model."""
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    model.eval()
    for batch in dataloader:
        inputs = {
            'input_ids': batch[0].to('cuda'),
            'attention_mask': batch[1].to('cuda'),
            'entity_pos': batch[3],
            'hts': batch[4],
        }
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(**inputs)
            pred = get_label(args, logits, num_labels=4)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    best_f1, _, best_f1_ign, _, p, r = official_evaluate(ans, args.data_dir, args.train_file, args.dev_file if tag == "dev" else args.test_file)
    return best_f1, {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_p": p * 100,
        tag + "_r": r * 100,
    }



def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--use_cl", type=int, default=1, help="Whether to use contrastive learning in the training.")
    parser.add_argument("--lambda_cl", default=0.01, type=float, help="Initial weight for the contrastive loss.")
    parser.add_argument("--adjust_lambda_cl", action="store_true", help="Dynamically adjust contrastive loss weight.")
    parser.add_argument("--lambda_cl_max", default=0.05, type=float, help="Max weight for the contrastive loss.")
    parser.add_argument("--lambda_cl_step", default=0.01, type=float, help="Step size for contrastive loss weight.")


    parser.add_argument("--cl_temperature", default=0.1, type=float, help="Temperature for contrastive loss.")
    parser.add_argument("--cl_margin", default=0.5, type=float, help="Margin for similarity in contrastive loss.")


    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_name", default="", type=str)
    parser.add_argument("--save_path", default="./checkpoint", type=str)
    parser.add_argument("--load_path", default="", type=str)

    # Tokenizer
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name if not the same.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name if not the same.")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="Maximum sequence length.")


    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Steps for gradient accumulation.")
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warmup ratio for learning rate.")
    parser.add_argument("--num_train_epochs", default=30, type=float, help="Total number of training epochs.")
    parser.add_argument("--evaluation_steps", default=-1, type=int, help="Steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
    parser.add_argument("--neg_lambda", type=float, default=1, help="Loss coefficient of negative samples.")
    parser.add_argument("--gamma", type=float, default=1, help="Focal loss gamma.")
    parser.add_argument("--proj_name", default="Re-DocRED", type=str, help="wandb project name.")
    parser.add_argument("--run_name", default="", type=str, help="wandb run name.")
    parser.add_argument("--loss_type", default="ATL", type=str, help="Loss type: ATL/balance_softmax/AFL/SAT/...")
    parser.add_argument("--neg_sample_rate", type=float, default=1, help="Negative sampling rate.")
    parser.add_argument("--margin", type=float, default=5, help="Hinge margin.")
    parser.add_argument("--nseed", nargs='+', type=int, default=[], help="Multiple seeds for repeated experiments.")
    parser.add_argument("--disable_log", action="store_true")
    parser.add_argument("--pos_only", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=1, help="CUDA device id.")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="Cache directory for pretrained models.")
    args = parser.parse_args()


    if args.load_path == "" and not args.disable_log:
        wandb.init(project=args.proj_name, name=args.run_name)
    else:
        wandb.init(project=args.proj_name, name=args.run_name, mode='disabled')
       
    torch.cuda.set_device(args.cuda_device)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    set_seed(args)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    train_features = read_docred(os.path.join(args.data_dir, args.train_file), tokenizer, args.max_seq_length)
    dev_features = read_docred(os.path.join(args.data_dir, args.dev_file), tokenizer, args.max_seq_length)
    test_features = read_docred(os.path.join(args.data_dir, args.test_file), tokenizer, args.max_seq_length)

    # 加载模型配置和模型
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    setattr(config, "cls_token_id", tokenizer.cls_token_id)
    setattr(config, "sep_token_id", tokenizer.sep_token_id)
    setattr(config, "transformer_type", args.transformer_type)

    base_model = AutoModel.from_pretrained(args.model_name_or_path, config=config, cache_dir=args.cache_dir)
    model = DocREModel(config, base_model, num_labels=args.num_labels)
    model.to(args.device)


    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)
    else:
        if args.nseed:
            for seed in args.nseed:
                args.seed = seed
                set_seed(args)
                best_dev_output = train(args, model, train_features, dev_features, test_features)
                print(f"Seed {seed}: Best dev output:", best_dev_output)
        else:
            best_dev_output = train(args, model, train_features, dev_features, test_features)
            print("Training complete. Best dev output:", best_dev_output)




if __name__ == "__main__":
    main()
