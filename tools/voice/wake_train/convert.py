"""Exports trained microWakeWord weights as the app's streaming int8 TFLite (same as model_train_eval's
--test_tflite_streaming_quantized path, without its test pass).

python3 -I convert.py <training.yaml> <weights.h5> <out.tflite> mixednet <model flags...>
"""
import argparse
import os
import shutil
import sys

from microwakeword import data as input_data
from microwakeword import mixednet, utils
from microwakeword.layers import modes
from microwakeword.model_train_eval import load_config


def main():
    cfg_path, weights, out = sys.argv[1:4]
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', default=cfg_path)
    sub = parser.add_subparsers(dest='model_name')
    mixednet.model_parameters(sub.add_parser('mixednet'))
    flags = parser.parse_args(sys.argv[4:])
    flags.training_config = cfg_path
    config = load_config(flags, mixednet)
    tmp = out + '.work'
    config['train_dir'] = tmp
    os.makedirs(tmp, exist_ok=True)
    data = input_data.FeatureHandler(config)
    model = mixednet.model(flags, shape=config['training_input_shape'], batch_size=1)
    model.load_weights(weights)
    utils.convert_model_saved(model, config, folder='stream_state_internal',
                              mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    utils.convert_saved_model_to_tflite(config, audio_processor=data,
                                        path_to_model=os.path.join(tmp, 'stream_state_internal'),
                                        folder=tmp, fname='model.tflite', quantize=True)
    shutil.move(os.path.join(tmp, 'model.tflite'), out)
    shutil.rmtree(tmp)
    print(out, os.path.getsize(out))


if __name__ == '__main__':
    main()
