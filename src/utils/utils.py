import logging
import os
from timm.utils import get_state_dict
from pathlib import Path
import torch
from .distributed import is_main_process, save_on_master

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        logging.warning("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logging.warning("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        logging.warning("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        logging.error('\n'.join(error_msgs))

def save_model(args, epoch, model, model_without_ddp, optimizer, save_path):
    to_save = {
        'epoch' : epoch,
        'model_state_dict': model_without_ddp.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
    }

    save_on_master(to_save, save_path)
    
    if is_main_process():
        _input = torch.randn(1, 3, args.input_size, args.input_size, device=args.device)
        export_dir = os.path.join(args.output_dir, "exported_models")
        onnx_path = os.path.join(export_dir, "model_onnx.onnx")
        torchscript_path = os.path.join(export_dir, "model_torchscript.pt")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        convert_to_torchscript(model, _input, torchscript_path)
        convert_to_onnx(model, _input, onnx_path)
        
def convert_to_torchscript(model, input_tensor, output_path, set_to_eval=True):
    if set_to_eval:
        model.eval()
    scripted_model = torch.jit.trace(model, input_tensor)
    scripted_model.save(output_path)
    logging.info(f"Model exported to Torchscript format at {output_path}")

def convert_to_onnx(model, input_tensor, output_path):
    torch.onnx.export(model, input_tensor, output_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    logging.info(f"Model exported to ONNX format at {output_path}")
    
def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if len(args.checkpoint) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.checkpoint = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        logging.info("Auto resume checkpoint: %s" % args.checkpoint)

    if args.checkpoint:
        if args.checkpoint.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.checkpoint, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        logging.info("Resume checkpoint %s" % args.checkpoint)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            logging.info("With optim & sched!")
