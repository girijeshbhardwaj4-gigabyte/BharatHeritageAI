
from PIL import Image, UnidentifiedImageError


def process_image(uploaded_file):

    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Convert image into RGB format
        image = image.convert("RGB")

        return image

    except UnidentifiedImageError:
        raise ValueError("Invalid image file. Please upload a valid image.")

    except Exception as error:
        raise ValueError(f"Unable to process image: {error}")
