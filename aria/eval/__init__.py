"""
Plan:
    
Embeddings experiments:
    ARIA-MIDI - Classification of genre classical vs jazz (vs other?)
    ARIA-MIDI - Classification musical time period (classical, baroque, ect...)
    ARIA-MIDI - Classification of top 5 composers
    ARIA-MIDI - Classification of top 5 pianists
    
    Pianist8 - Classification -- should work
    VGMIDI - Classification -- Probably won't work as it's multi-track
    WikiMT - Classification -- No idea

Ablation comparisons:
    Frozen classical embeddings (both mean and last-token from pretrained model)
    Aria finetuned on these specific classification tasks (define new <CLS> token) - TODO
    Aria trained from scratch on these specific classification tasks (define new <CLS> token) - TODO
    Aria finetuned with contrastive learning - TODO
    Aria trained with contrastive learning without next-token pretraining - TODO (Maybe skip)
    
Other model comparisons:
    Clamp2 or Clamp3
    MusicBERT

"""
