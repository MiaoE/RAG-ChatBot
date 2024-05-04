# pip library
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', '-a', required=False, help="Action")
    arguments = parser.parse_args()
    print('Argument for action: ' + arguments.action)
