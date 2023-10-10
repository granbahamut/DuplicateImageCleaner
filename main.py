# Main imports
import multiprocessing
import random
import cv2
import numpy as np
import pathlib
from multiprocessing import Process, current_process
import matplotlib.pyplot as plt

path = "X:\\SomePathHere"


def adjust_images(img_a, img_b):
    """Convert the images passed as parameters to grayscale. Also resizes de images to have them both
    with similar dimensions.

    Args:
        img_a : InputArray, required
            Image to be converted to grayscale.
        img_b : InputArray, required
            Image to be converted to grayscale and to be resized to the dimensions of img_a.

    Returns:
        The 2 images processed as grayscale and with the same dimensions.
    Raises
        None.
    """

    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    h, w = img_a.shape
    img_b = cv2.resize(img_b, (w, h))
    return img_a, img_b


def mse(image_a, image_b):
    """The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
    NOTE: the two images must have the same dimension return the MSE, the lower the error, the more "similar".

    Args:
        image_a : InputArray, required
            First image to compare.
        image_b : InputArray, required
            Second image to be compared with image_a.

    Returns:
        The MSE as a float value indicating the similarity of the 2 images.
        The lower the value, the more similar the images are.
    Raises
        None.
    """
    w = int(image_b.shape[1])
    h = int(image_b.shape[0])
    dim = (w, h)
    resized = cv2.resize(image_a, dim, interpolation=cv2.INTER_AREA)
    err = np.sum((resized.astype("float") - image_b.astype("float")) ** 2)
    err /= float(resized.shape[0] * resized.shape[1])
    return err


def show_mse(img1, img2):
    """This just prints the MSE on console and allows to see the images on a frame to look at them.
    Just for testing purposes.

    Args:
        img1 : InputArray, required
            First image to compare.
        img2 : InputArray, required
            Second image to be compared with img1.

    Returns:
        None.
    Raises
        None.
    """
    error = mse(img1, img2)
    print("Image matching Error between {} and {}: {}%".format(img1, img2, error))
    plt.imshow(img1)
    # cv2.imshow("difference", diff)
    plt.imshow(img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Just a utility.

    Args:
        lst : Array, required
            Array of object to be spliced.
        n : InputArray, required
            amount of chunks to splice de main array.

    Returns:
        The same list but in a 2 dimensional array, where each array is the size of the main list
        divided by the amount of chunks.
    Raises
        None.
    """
    try:
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    except ValueError:
        print("No files were found to partitioning into chunks of files.")
    finally:
        exit(2)


def run_multi_comparison(base_path):
    """Takes the amount of processors available on a machine where this script is executing, takes each chunk of
    images from base_ath and put it on a dedicated process to compare each image with the base image list.

    Args:
        base_path : String, required
            Path where the images to compare are stored.

    Returns:
        None.
    Raises
        None.
    """
    print("Reading {} for duplicates...".format(base_path))
    image_files_db = []
    try:
        image_files_db = [f for f in pathlib.Path(base_path).iterdir()
                          if f.is_file() and f.suffix == ".png" or f.suffix == ".jpeg" or f.suffix == ".jpg"]
        if image_files_db.count == 0:
            raise FileExistsError()
    except FileNotFoundError:
        print("Path {} not found.", base_path)
    except FileExistsError:
        print("There's no files on {} that are images.", base_path)
    except OSError:
        print("Invalid path given, correct it and thy again.")

    processing_threads = 4  # multiprocessing.cpu_count()
    print("Using {} processors from {} for this execution.".format(processing_threads, multiprocessing.cpu_count()))
    files_per_processor = (len(image_files_db) // processing_threads)
    file_chunk_list = chunks(image_files_db, files_per_processor)
    print("Amount of image files in BD: {}. "
          "Processors: {}. "
          "Files per processor: {}.".format(len(image_files_db), processing_threads, files_per_processor))
    procs = []
    for image_chunk in file_chunk_list:
        proc = Process(target=compare_images, args=(image_chunk, image_files_db,))
        proc.name = str(random.randint(1, 500000))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def compare_images(image_comparison_set, img_db):
    """Compares a list of images with another image list, prints on console the MSE of 2 images that are too similar.
    The MSE is set to 10%, so if 2 images are 90% similar, both paths of the images will be printed.

    Args:
        image_comparison_set : Array, required
            Array of images to be compared with img_db.
        img_db : Array, required
            Image set to be used as a comparison base.

    Returns:
        None.
    Raises
        None.
    """
    try:
        for img_from_db in img_db:
            img_1 = cv2.imread(str(img_from_db))
            for img_from_set in image_comparison_set:
                img_2 = cv2.imread(str(img_from_set))
                error = mse(img_1, img_2)

                # If MSE is equal or less than 10, then the image is likely duplicate
                if 0.0 <= error <= 10.0:
                    # If the image has the same name, omit it from the comparison
                    if str(img_from_db) != str(img_from_set):
                        pr = current_process()
                        # global duplicated_count
                        # duplicated_count += 1
                        print("[{}] Image matching MSE between {} and {} is: {}%."
                              .format(pr, str(img_from_db), str(img_from_set), error))
    except AttributeError:
        print("AttributeError caught, will continue...")
    except NameError:
        print("Some name was not defined, duplicated_count?, will continue...")


if __name__ == "__main__":
    run_multi_comparison(path)
