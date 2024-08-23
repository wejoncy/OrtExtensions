
import os
print(os.getpid())
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import transformers
import onnxruntime
import numpy as np
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from typing import List, Optional, Tuple
import random

import pickle
class OnnxAddpadding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hd, mask) -> torch.Tensor:
        output = torch.zeros(mask.shape+(hd.shape[-1],), dtype=hd.dtype, device=hd.device)
        seq_len = mask.sum(-1)
        print(seq_len)
        pre_len = 0
        for i in range(mask.shape[0]):
            output[i, :seq_len[i]] =  hd[0,pre_len:pre_len+seq_len[i]]
            pre_len += seq_len[i]
        return output


    @staticmethod
    def symbolic(g: torch.Graph, hd, mask) -> torch.Value:
        return g.op("vllm.ort.ext::AddPadding",hd, mask)

class PagedAttentionNet(torch.nn.Module):
    def __init__(self,num_heads=32, num_kv_heads=32, scale=128**-0.5,sliding_window=None,sin_cos_cache=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = 128
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window

    def forward(self,hd: torch.Tensor, mask: torch.Tensor,):
        return OnnxAddpadding.apply(hd,mask)

def create_custom_onnx():
    X = make_tensor_value_info("input_ids", TensorProto.INT64, [None,None])
    Y = make_tensor_value_info("attention_mask", TensorProto.INT64, [None, None])
    Z = make_tensor_value_info("Z", TensorProto.INT64, [None, None])
    graph = make_graph(
        [make_node("RemovePadding", ["input_ids","attention_mask"], ["Z"], "example_cus", domain="vllm.ort.ext")],
        "custom_op",
        [X, Y],
        [Z],
    )
    onnx_model = make_model(
        graph,
        opset_imports=[make_opsetid("", 17), make_opsetid("vllm.ort.ext", 1)],
        ir_version=9,
    )
    check_model(onnx_model)
    with open("custom_op_test.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    torch_module = PagedAttentionNet()
    torch_module.eval()
    seq_dim=1
    layer_num=1
    onnx_dynamic_axes = {"hd": {1: "seq_len"}, "mask": {0:"batc", 1: "maxs"}}

    onnx_inp_names = ["hd", "mask"]
    onnx_inp_names = tuple(onnx_inp_names)

    onnx_out_names = ("last_hidden_state",)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained('../Mistral-7B-v0.1-GPTQ')
    max_seq_length = 256
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    batch_text = ["Hello, my dog is cute", "last time, I went to the zoo", "I am a student, and ", "I am a teacher", "I am a doctor"]
    device = torch.device("cuda")
    inputs = tokenizer(batch_text, return_tensors="pt", max_length=max_seq_length, padding="max_length", truncation=True)
    attention_mask_onnx = inputs["attention_mask"].to(device)

    onnx_inputs = [torch.rand((1, attention_mask_onnx.sum(), 4096,), dtype=torch.half, device=device), attention_mask_onnx]
    t_out = torch_module(*onnx_inputs)
    torch.onnx.export(model=torch_module, args=tuple(onnx_inputs), f=str('custom_addpadding_op_test.onnx'), verbose=False, opset_version=17,
                      input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)

    custom_op_library_path = 'libcustom_op_library.so'
    session_options = onnxruntime.SessionOptions()
    session_options.register_custom_ops_library(custom_op_library_path)
    ort_sess = onnxruntime.InferenceSession('custom_addpadding_op_test.onnx', sess_options=session_options, providers=['CUDAExecutionProvider'])
    onnx_inputs = {"hd":onnx_inputs[0].cpu().numpy(), "mask":onnx_inputs[1].cpu().numpy()}
    ort_out = ort_sess.run(None, onnx_inputs)
    print(ort_out[0] - t_out.cpu().numpy(), (ort_out[0] - t_out.cpu().numpy()).sum())
    print('------------------')




create_custom_onnx()
custom_op_library_path = 'libcustom_op_library.so'
session_options = onnxruntime.SessionOptions()
session_options.register_custom_ops_library(custom_op_library_path)
ort_sess = onnxruntime.InferenceSession('custom_op_test.onnx', sess_options=session_options, providers=['CUDAExecutionProvider'])


tokenizer = transformers.AutoTokenizer.from_pretrained('../Mistral-7B-v0.1-GPTQ')

max_seq_length = 256

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
batch_text = ["Hello, my dog is cute", "last time, I went to the zoo", "I am a student, and ", "I am a teacher", "I am a doctor"]
inputs = tokenizer(batch_text)#, return_tensors="pt", max_length=max_seq_length, padding="max_length", truncation=True)

device = torch.device("cuda")
inputs_ids_onnx = torch.tensor(sum(inputs["input_ids"], [])).to(device)
attention_mask_onnx = torch.tensor(sum([list(range(sum(i))) for i in inputs["attention_mask"]],[])).to(device)


inputs = tokenizer(batch_text, return_tensors="pt", max_length=max_seq_length, padding="max_length", truncation=True)
inputs=inputs.to(device)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']


onnx_inputs = {"input_ids":input_ids.cpu().numpy(), "attention_mask":attention_mask.cpu().numpy()}

ort_out = ort_sess.run(None, onnx_inputs)
print(ort_out[0]-inputs_ids_onnx.cpu().numpy())
