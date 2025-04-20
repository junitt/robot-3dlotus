import os
import argparse

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from flask import Flask, request

from genrobo3d.agents.ThreeDLotus import ThreeDLotusActioner


def main(args):
    app = Flask(__name__)
    
    actioner = ThreeDLotusActioner()

    @app.route('/predict', methods=['POST'])
    def predict():
        '''
        batch is a dict containing:
            taskvar: str, 'task+variation'
            episode_id: int
            step_id: int, [0, 25]
            instruction: str
            obs_state_dict: observations from genrobo3d.rlbench.environments.RLBenchEnv 
        '''
        data = request.data
        batch = msgpack_numpy.unpackb(data)

        action = actioner.predict(**batch)

        action = msgpack_numpy.packb(action)
        return action
    
    app.run(host=args.ip, port=args.port, debug=args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Actioner server')
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=13000)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--model', choices=['3dlotus', '3dlotusplus'], default='3dlotusplus')
    args = parser.parse_args()
    main(args)
