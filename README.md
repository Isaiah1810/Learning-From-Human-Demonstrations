# Learning From Human Demonstrations



## Installation

To set up the project environment, install the required dependencies using pip:

```bash
pip install -r environments.txt
```

## Usage

### Converting Dataset to Tokenized Dataset

To convert your dataset to a tokenized format:

```bash
python scripts/tokenize_data.py <path> <output_path> <image_size>
```

Parameters:
- `<path>`: Path to the input dataset
- `<output_path>`: Path where tokenized dataset will be saved
- `<image_size>`: Size to scale images to (in pixels)

### Training Latent Action Model

To train the latent action model:

```bash
accelerate launch latent_action_model/train_latent_action.py
```

### Training Latent Action Predictor Model

To train the latent action predictor model:

```bash
accelerate launch train.py --config <config path>
```

Parameters:
- `<config path>`: Path to the configuration file

