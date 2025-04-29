## Running the Melanoma Classifier

Full description available in our technical documentation [./documentation/technical_documentation.pdf]

### Training the Model

There are two ways to train our melanoma classification model:

#### Option 1: Config-based Training

To run the experiment and create our best model using a configuration file:

```bash
python melanoma_train.py --config "./configs/best_run.yaml"
```

#### Option 2: Script with Predefined Parameters

We provide a convenient bash script that automatically detects GPU availability and sets parameters:

```bash
bash train_config.sh
```

### Evaluating the Model

To evaluate a trained model:

   - Edit the configuration file by setting the 'test' field to 'true'
   - Run the evaluation command:

```bash
python melanoma_train.py --config "./configs/best_run.yaml" --checkpoint <PATH_TO_WEIGHTS>
```
 
Or you can modify parametes in th training script:


```bash
python melanoma_train.py \
  --config "configs/dino_vit_small.yaml" \
  --data_path "/path/to/dataset" \
  --skin_color_csv "/path/to/metadata.csv" \
  --device "cuda" \
  --num_workers 4 \
  --log_dir "./melanoma_logs" \
  --output_dir "./melanoma_classifier_output"
```