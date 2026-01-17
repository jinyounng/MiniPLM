"""Processing data"""
import os
import json
import numpy as np
import random
import torch
import time
import multiprocessing
from utils import print_args, PAD_EOS_MODELS, BOS_MODELS
from data_utils import ChunkedDatasetBuilder, best_fitting_dtype
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args, add_peft_args
import argparse
from transformers import AutoTokenizer


random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        if self.args.model_type in PAD_EOS_MODELS:
            Encoder.tokenizer.pad_token = Encoder.tokenizer.eos_token

    def encode(self, id_with_json_line):
        doc_id, json_line = id_with_json_line
        line = json.loads(json_line)
        doc = line["text"]
        label = line["meta"]["pile_set_name"]
        tokens = Encoder.tokenizer.encode(
            doc, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        # del doc
        # del line

        return tokens, doc_id, label, len(doc)


class Writer():
    def __init__(self,
                 args,
                 output_path,
                 tokenizer,
                 builder,
                 label_index,
                 end_sent_mask,
                 rt_token_mask,
                 dtype):
        self.args = args
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.label = label_index
        if self.args.model_type in BOS_MODELS:
            self.chunk_tokens_buffer = [tokenizer.bos_token_id]
        else:
            self.chunk_tokens_buffer = []
        self.end_sent_mask = end_sent_mask
        self.rt_token_mask = rt_token_mask
        self.builder = builder
        self.sid = 0
        self.padded_token_num = 0
        self.dtype = dtype
    
    def check_sent_end(self, tokenizer, i, new_chunk, chunk_tokens_buffer):
        model_type = self.args.model_type
        if model_type in BOS_MODELS:
            tokens = (new_chunk+chunk_tokens_buffer[1:2])[i:i+2]
        else:
            tokens = (new_chunk+chunk_tokens_buffer[:1])[i:i+2]
        s = tokenizer.decode(tokens)
        return len(tokens) == 1 or (" " in s)
    
    def add_tokens(self, doc_tokens, lid):
        self.chunk_tokens_buffer.extend(doc_tokens)
        n = 0
        while len(self.chunk_tokens_buffer) >= self.args.max_length:
            new_chunk = self.chunk_tokens_buffer[:self.args.max_length]
            if self.args.model_type in BOS_MODELS:
                self.chunk_tokens_buffer = [self.tokenizer.bos_token_id] + \
                    self.chunk_tokens_buffer[self.args.max_length:]
            else:
                self.chunk_tokens_buffer = self.chunk_tokens_buffer[self.args.max_length:]

            for i in range(len(new_chunk)-1, -1, -1):
                if (new_chunk[i] in [self.tokenizer.eos_token_id]) or \
                    (self.rt_token_mask[new_chunk[i]]) or \
                    (self.end_sent_mask[new_chunk[i]] and self.check_sent_end(self.tokenizer, i, new_chunk, self.chunk_tokens_buffer)): # check if the end is fake
                    # 1. Who are you? I am the D.A. and he is //Bat Man. -> Who are you? // I am the D.A. and he is Bat Man.
                    # 2. Who are you? I am the D.//A. -> Who are you? // I am the D.A.
                    incomplete_sent = new_chunk[i+1:]
                    # new_chunk = new_chunk[:i+1] + [tokenizer.pad_token_id] * (args.max_length - (i+1))
                    new_chunk = new_chunk[:i+1]
                    if self.args.model_type in BOS_MODELS:
                        self.chunk_tokens_buffer = self.chunk_tokens_buffer[:1] + incomplete_sent + self.chunk_tokens_buffer[1:]
                    else:
                        self.chunk_tokens_buffer = incomplete_sent + self.chunk_tokens_buffer
                    self.padded_token_num += self.args.max_length - (i+1)

                    break
            
            if self.args.model_type in BOS_MODELS:
                assert new_chunk[0] == self.tokenizer.bos_token_id
            if len(new_chunk) <= 1:
                continue
            assert len(new_chunk) <= self.args.max_length
            
            self.sid += 1
            n += 1
            if n > 500 and (n % 100 == 0):
                print_and_save(f"Constructing {n} chunks from a document, chunk len: {len(new_chunk)}", self.output_path)
                print_and_save(f"Chunk: {new_chunk}", self.output_path)
            if n > 2000:
                if self.args.model_type in BOS_MODELS:
                    self.chunk_tokens_buffer = [self.tokenizer.bos_token_id]
                else:
                    self.chunk_tokens_buffer = []
                break
            self.builder.add_np_item(np.array(new_chunk, dtype=self.dtype))


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_peft_args(add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser)))))
    args = parser.parse_args()

    return args


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(s + "\n")


def get_ent_sent_infos(args, tokenizer):
    with open(os.path.join(args.base_path, "tools", "process_data", f"end_sent_token_{args.model_type}.json"), "r") as f:
        end_sent_token = json.load(f)
    end_sent_mask = np.zeros(len(tokenizer), dtype=bool)
    for token in end_sent_token:
        end_sent_mask[token] = True
    rt_token_mask = np.zeros(len(tokenizer), dtype=bool)
    for token in end_sent_token:
        if "\n" in tokenizer.decode([token]):
            rt_token_mask[token] = True

    return end_sent_mask, rt_token_mask


def main():
    args = get_args()
    random.seed(args.seed)
    output_path = args.save

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    dtype = best_fitting_dtype(len(tokenizer))

    output_path = os.path.join(output_path, args.model_type + "-" + str(args.max_length))
    os.makedirs(output_path, exist_ok=True)
        
    print_and_save(f"Tokenizer size: {len(tokenizer)}. Using dtype: {dtype}", output_path)
    
    if args.model_type in PAD_EOS_MODELS:
        tokenizer.pad_token = tokenizer.eos_token
    
    print_and_save(f"Output path: {output_path}", output_path)
    # os.makedirs(output_path.replace("processed_data_1", "processed_data"), exist_ok=True)
    
    print_args(args)    
    with open(os.path.join(output_path, "args_get_paragraph.json"), "w") as f:
        json.dump(vars(args), f)
    
    end_sent_mask, rt_token_mask = get_ent_sent_infos(args, tokenizer)

    builder = ChunkedDatasetBuilder(
            base_path=args.base_path,
            output_path=output_path,
            dtype=dtype,
            split="data",
            do_shuffle=True)

    startup_start = time.time()

    print("input path", args.data_dir)
    print("Output path:", output_path)

    sid, lid = 0, 0
    log_bytes_processed, log_doc_proccessed = 0, 0
    padded_token_num = 0

    with open(os.path.join(args.base_path, "tools", "process_data", f"domain_labels.json"), "r") as f:
        domain_labels = json.load(f)
    writers = {}

    encoder = Encoder(args)
    pool = multiprocessing.Pool(
        args.data_process_workers, initializer=encoder.initializer)
    
    global_start = time.time()

    if os.path.exists(os.path.join(args.base_path, "tools", "process_data", f"files_names.json")):
        with open(os.path.join(args.base_path, "tools", "process_data", f"files_names.json"), "r") as f:
            files_names = json.load(f)
    else:
        files_names = os.listdir(args.data_dir)
        random.shuffle(files_names)
        with open(os.path.join(args.base_path, "tools", "process_data", f"files_names.json"), "w") as f:
            json.dump(files_names, f)

    print(files_names)

    print_and_save(f"Shard start: {args.shard_start}. Shard end: {args.shard_end}.", output_path)
    files_names = files_names[args.shard_start:args.shard_end]

    proc_start = global_start
        
    for fid, file_name in enumerate(files_names):
        print_and_save(f"Processing {file_name}. {fid}/{len(files_names)}", output_path)
        input_file = os.path.join(args.data_dir, file_name)
        fin = open(input_file)

        # use the tokenizer to encode the sentences
        encoded_docs = pool.imap(encoder.encode, enumerate(fin), 20)

        for doc_tokens, doc_id, label, bytes_processed in encoded_docs:
            lid += 1
            log_bytes_processed += bytes_processed
            log_doc_proccessed += 1
            if label in writers:
                writer = writers[label]
            else:
                label_idx = domain_labels[label]
                writer = Writer(args, output_path, tokenizer, builder, label_idx, end_sent_mask, rt_token_mask, dtype)
                writers[label] = writer

            writer.add_tokens(doc_tokens, lid)
            sid = sum([writers[k].sid for k in writers])
            padded_token_num = sum([writers[k].padded_token_num for k in writers])
            if lid % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = log_bytes_processed / elapsed / 1024 / 1024
                ds = log_doc_proccessed / elapsed
                tokens = (sid * args.max_length - padded_token_num) / 1e9

                s = f"Processed {lid} documents. {sid} chunks. {tokens:.4f}B tokens. " + \
                    f"Padding fraction: {padded_token_num / (sid * args.max_length):.4f}." + \
                    f"({ds:.2f} docs/s, {mbs:.4f} MB/s). Total Time: {current - global_start} s."

                print_and_save(s, output_path)

                log_bytes_processed, log_doc_proccessed = 0, 0
                proc_start = current

            if builder.ofid >= args.max_shard_num:
                break

        fin.close()
        fin = None
        
        if builder.ofid >= args.max_shard_num:
            break

    builder.finalize()

    sid = sum([writers[k].sid for k in writers])
    padded_token_num = sum([writers[k].padded_token_num for k in writers])

    # summarize
    print_and_save(f"Total time: {time.time() - startup_start}.", output_path)
    print_and_save(f"Total processed paragraphs: {sid}.", output_path)
    total_tokens = sid * args.max_length - padded_token_num
    print_and_save(f"Total tokens: {total_tokens / 1e9:.4f}B", output_path)
    print_and_save(f"Total padding fraction: {padded_token_num / (sid * args.max_length)}.", output_path)

    if fin is not None:
        fin.close()

    pool.terminate()
    pool.close()
    pool.join()
    pool = None

if __name__ == '__main__':
    main()
