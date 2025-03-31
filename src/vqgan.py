from OmniTokenizer import OmniTokenizer_VQGAN



class ImageTokenizer():
    def __init__(self, model_path: str):
        self.vqgan = OmniTokenizer_VQGAN.load_from_checkpoint("./imagenet_k600.ckpt", strict=False, weights_only=False)
        self.vqgan.eval()

    def encode(self, images, include_encodings=False, keep_dims=False):
        data = data.cuda()

        embeddings, encodings = self.vqgan.encode(data, True, True)

        self.original_shape = embeddings.shape

        if not keep_dims:
            embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1).permute(0, 2, 1)
        
        if include_encodings:
            return embeddings, encodings
        
        return embeddings
    
    def decode(self, embeddings):

        if embeddings.dim == 3:
            embeddings = embeddings.reshape(self.original_shape)

        encodings = self.vqgan.codebook.embeddings_to_encodings(embeddings)
        recons = self.vqgan.decode(encodings, True)

        return recons