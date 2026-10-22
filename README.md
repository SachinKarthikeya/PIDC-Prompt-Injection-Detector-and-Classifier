# 🦠 PIDC - Prompt Injection Detector and Classifier

An efficient NLP based detection system system which analyzes user prompts/texts to detect any prompt injection/jailbreak/system override attempts that can be performed on Large Language Models. This can help systems to detect any malicious attacks that happen on LLMs and take necessary measures before the data gets compromised.

## 🚀 Features

- **HuggingFace**: Downloaded the dataset which contained 6,274 rows
- Output section is divided into three different coluns:
    - **Label**: 0 for safe, 1 for unsafe
    - **Category**: Contains 31 different types of attacks based on the texts
    - **Severity**: 4 types i.e., low, medium, high, critical 
- **Scikit-learn**: Split the dataset and evaluate the model performance
- **Transformers**: Tokenized the texts and loaded the pre-trained DistilBERT model for training
- **PyTorch**: Define the encoder with classification heads and add additional layers like Dropout and Softmax layers
- **JSON**: Produce the model outputs in JSON-structured format with reported:
    - Detected Label
    - Probability of the Detected Label
    - Detected Category
    - Probability of the Detected Category
    - Detected Severity
    - Probability of the Detected Severity

## 📄 Workflow

- The system analyzes the prompt coming from the user
- The prompt/text gets tokenized before it gets passed through
- The trained model along with the saved category and severity mapping files are loaded
- The model detects and classfies the text using the three classification heads:
    - Label Head
    - Category Head
    - Severity Head
- THe output is shown in a structured JSON format for better understanding

## 🧰 Tech Stack

- **Backend**: Pandas, Numpy, Scikit-learn, Pytorch, Transformers, Json
- **Dataset**: HuggingFace