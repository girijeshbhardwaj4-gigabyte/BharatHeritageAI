
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

MODEL_NAME = "Salesforce/blip-vqa-base"


class HeritageVQA:

    def __init__(self):

        # Select GPU if available, otherwise use CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model on: {self.device}")

        # Load pretrained processor and model
        self.processor = BlipProcessor.from_pretrained(MODEL_NAME)

        self.model = BlipForQuestionAnswering.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print("BLIP VQA model loaded successfully!")

    def answer_question(self, image, question):

        # Process image and question
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )

        # Move inputs to selected device
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Generate answer
        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40
            )

        # Convert generated tokens into text
        answer = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer
