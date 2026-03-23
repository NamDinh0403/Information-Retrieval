import timm

def download_vit_weights(model_name='vit_base_patch16_224'):
    """
    Downloads the pre-trained weights for the Vision Transformer model
    from huggingface/timm registry to the local cache.
    """
    print(f"Downloading pre-trained weights for {model_name}...")
    
    # Creating the model with pretrained=True triggers the download 
    # to the local torch/huggingface cache.
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    print("Model weights downloaded and cached successfully!")

if __name__ == "__main__":
    download_vit_weights()
