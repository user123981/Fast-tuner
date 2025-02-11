#re-train OCTMAE-base with your scheme (the original model, best one) and re-eval linear probing just to be sure
# also save the text encoder from that as well for zero shot



# for zero-shot classification, 
    #pass a kermany image through the ViT backbone, and 4 prompts individually (one for each class in kermany) through BERT,

# calculate the similarity between the image embedding and each text embedding

# then the highest similarity is the predicted class

# compare with fine-tuned BERT and original BERT