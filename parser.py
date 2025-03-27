import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='APO4Rec')
    parser.add_argument('--reward_func', 
                        type=str,
                        default='ndcg',
                        help='how the reward is calculated, options: ndcg')
    parser.add_argument('--model', 
                        type=str,
                        default='deepseek-ai/DeepSeek-R1-Turbo',
                        help='which model as recommender, options: meta-llama/Meta-Llama-3.1-70B-Instruct')
    # parser.add_argument('--model', 
    #                     type=str,
    #                     default='llama3-8b-8192',
    #                     help='options: llama3-8b-8192, llava-v1.5-7b-4096-preview')
    parser.add_argument('--seed', 
                        type=int,
                        default=42,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=20,
                        help='options: 10, 20')
    parser.add_argument('--dataset', 
                        type=str,
                        default='bundle',
                        help='use which datset: bundle/games/ml-1m')
    parser.add_argument('--train_num', 
                        type=int,
                        default=50,
                        help='options: 50,150')
    parser.add_argument('--batch_size', 
                        type=int,
                        default=16,
                        help='options: 16,32')

    args = parser.parse_args()
    
    return args