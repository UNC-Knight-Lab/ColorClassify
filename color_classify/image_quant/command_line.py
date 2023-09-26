from .image_quantification import image_quantification
import os
import argparse


def main():
    # Use current working directory as default input/output folder
    cwd = os.getcwd()
    input_folder = cwd

    # Parse arguments for input/output folder
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Full path to input folder where image .tiff data is.')
    args = parser.parse_args()

    if args.i:
        input_folder = args.i
    
    new_dirs = ['annotated_images', 'image_RGB']

    for directory in new_dirs:
        path = os.path.join(input_folder, directory)
        os.mkdir(path)
        print("Directory '% s' created" % directory)

    # Call image_quantification function
    image_quantification(input_folder)

if __name__ == "__main__":
    main()