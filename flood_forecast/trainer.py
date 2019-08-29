import argparse 

def main():
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-d", "--dataset", help="Path to river flow csv file")
    parser.add_argument("-m", "--model", help="Model you want to use for training")
    parser.add_argument("-t", "--task", help="The task you want to train the model for")
if __name__ == "__main__":
    main()