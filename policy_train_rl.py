import argparse
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from stable_baselines3.common.env_util import make_vec_env
from rl_env import Sim
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ENVIROMENT_CLASSES = {
    'real': Sim
}


def make_env(use_gui, discrete, env_class_name, target_number_of_objects, reset_interval, classification_model):
    env_class = ENVIROMENT_CLASSES[env_class_name]
    return Monitor(env_class(gui=use_gui, discrete=discrete, number_of_objects=target_number_of_objects, reset_interval=reset_interval, classification_model=classification_model))


def main():
    parser = argparse.ArgumentParser(description='Gripper DRL.')
    parser.add_argument('--jobs', default=1, type=int, help='Number of parallel simulations')
    parser.add_argument('--algorithm', default='PPO', type=str, help='Algorithm')
    parser.add_argument('--gui', default=False, action='store_true')
    parser.add_argument('--model', default='', type=str, help='Path to the model')
    parser.add_argument('--play-only', default=False, action='store_true')
    parser.add_argument('--environment', default='real', type=str)
    parser.add_argument('--n_steps', default=None, type=int)
    parser.add_argument('--target_number_of_objects', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--reset_interval', default=1, type=int)

    # sac
    parser.add_argument('--discrete', default=True, action='store_true')
    parser.add_argument('--gradient_steps', default=1, type=int)
    parser.add_argument('--train_freq', default=1, type=int)
    parser.add_argument('--buffer_size', default=500_000, type=int)
    parser.add_argument('--learning_starts', default=1000, type=int)
    args = parser.parse_args()

    # classification_model = torch.load('net.pth').cpu()
    classification_model = torch.load('./best_classifier.pth',
                                      map_location=torch.device('cpu'))
    # classification_model = None
    env = make_vec_env(lambda: make_env(args.gui, args.discrete, args.environment,
                                        args.target_number_of_objects,
                                        args.reset_interval, classification_model),
                       n_envs=args.jobs, vec_env_cls=SubprocVecEnv)

    # Load DRL algorithm
    drl_algorithm_classes = {
        'PPO': stable_baselines3.PPO,
        'SAC': stable_baselines3.SAC
    }
    drl_algorithm_class = drl_algorithm_classes[args.algorithm]

    # Initialize model
    if args.model != '':
        # Load model
        model = drl_algorithm_class.load(
            args.model, env=env, learning_rate=args.lr, n_steps=args.n_steps)
        print('load model successful')
    else:
        algorithm_args = {}
        # algorithm_args['policy_kwargs'] = dict(
        #     features_extractor_class=CustomCombinedExtractor
        # )
        policy = 'MultiInputPolicy'

        if args.algorithm == 'SAC':
            algorithm_args['buffer_size'] = args.buffer_size
            algorithm_args['gradient_steps'] = args.gradient_steps
            algorithm_args['train_freq'] = args.train_freq
            algorithm_args['learning_starts'] = args.learning_starts

        if args.n_steps is not None:
            algorithm_args['n_steps'] = args.n_steps

        # Model from scratch
        model = drl_algorithm_class(
            policy,
            env,
            batch_size=1024,
            learning_rate=args.lr,
            verbose=1,
            tensorboard_log='logs/tb/all3',
            **algorithm_args
        )

    # Play or learn
    if args.play_only:
        # Play
        number_of_episodes = 0
        number_of_successes = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            print(action)
            if done:
                # break
                number_of_episodes += 1
                if reward >= 10:
                    number_of_successes += 1
                print(
                    f'Success picks {number_of_successes}/{number_of_episodes} ({(number_of_successes/number_of_episodes) * 100}%)')
                obs = env.reset()
    else:
        # Learn
        model.learn(
            total_timesteps=100_000_000_000_000,
            callback=[
                CheckpointCallback(
                    save_freq=max(args.n_steps*4, 1),
                    save_path='./logs/models/all3',
                ),
                # CustomEvalCallback(env, target_number_of_objects=args.target_number_of_objects,
                # workspace_radius=args.workspace_radius),
            ]
        )
        model.save('final')


if __name__ == '__main__':
    main()
