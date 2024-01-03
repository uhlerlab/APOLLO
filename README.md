# APOLLO

## Application to paired scRNA-seq and scATAC-seq
### APOLLO training
#### step 1 latent optimization
train_lord_randNoise_sharedRecon_shareseq_filter_bce_morefilter.ipynb  
#### step 2 inference
train_lord_randNoise_sharedRecon_shareseq_morefilter_reverse_bce.ipynb

### Cell type classification
train_clf_lord_shareseq_celltype_bce_morefilter.ipynb

### Interpretation of partially shared latent spaces
#### Genes or gene ontology terms with significant changes along each principal component of the latent spaces
plot_lord_bce_pca_sampling.ipynb - identify differentially expressed genes or peaks along each principal component of the shared or modality-specific latent spaces
plot_lord_bce_pca_sampling_withAnnotations_curve.ipynb - plot differentially expressed genes or peaks along each principal component of the shared or modality-specific latent spaces
plot_lord_bce_pca_sampling_withAnnotations.ipynb - plot the enriched gene ontology terms of the genes or peaks represented by the shared or modality-specific latent spaces

### Preprocessing of scATAC-seq data
preprocess_shareseq.ipynb

## Application to paired chromatin and protein images
### APOLLO training
#### step 1 latent optimization
train_cnnvae_splitChannels_conditional_lord_randNoise_bce.ipynb - BCE loss used for reconstruction  
train_cnnvae_splitChannels_conditional_lord_randNoise.ipynb - MSE loss used for reconstruction  
#### step 2 inference
train_cnnvae_splitChannels_conditional_lord_randNoise_reverse_bce.ipynb - inference step for the model trained with BCE loss   
train_cnnvae_splitChannels_conditional_lord_randNoise_reverse.ipynb - inference step for the model trained with MSE loss

### APOLLO without modality-specific latent spaces
train_cnnvae_splitChannels_conditional_lord_randNoise_fullyJoint.ipynb - step 1, latent optimization  
train_cnnvae_splitChannels_conditional_lord_randNoise_reverse_fullyJoint.ipynb - step 2, inference

### APOLLO trained with one-step training as an autoencoder
train_cnnvae_splitChannels_conditional.ipynb

### Phenotype classification using real images, reconstructed images from full latent space, reconstructed images from shared latent space, or protein images predicted from chromatin
plot_Clf_conditions_sampling.ipynb - plot results  
train_clf_conditions_c2c_fullrecon_sampling.ipynb - train classifiers using reconstructed chromatin images from the full latent space  
train_clf_conditions_c2c_sharedrecon_sampling.ipynb - train classifiers using reconstructed chromatin images from the shared latent space  
train_Clf_conditions_c2p_sampling.ipynb - train classifiers using protein images predicted from chromatin  
train_clf_conditions_originalImg_chromatin_sampling.ipynb - train classifiers using the original chromatin images  
train_clf_conditions_originalImg_sampling.ipynb - train classifiers using the original protein images  
train_Clf_conditions_p2p_fullrecon_sampling.ipynb - train classifiers using reconstructed protein images from the full latent space  
train_Clf_conditions_p2p_sharedRecon_sampling.ipynb - train classifiers using reconstructed protein images from the shared latent space  

### Interpretation of partially shared latent spaces of paired chromatin and protein images
#### Manually selected chromatin and protein morphological features
getNMCO_allFeatures.ipynb - preprocess  
getNMCOgroups.ipymb - group chromatin features by correlation and selecting one representative feature for each group  
getNMCOgroups_protein.ipymb - group protein features by correlation and selecting one representative feature for each group  
#### Plotting examples of chromatin or protein images along each PC
plot_examples_centerPCs_percentiles_noHeldOut.ipynb
#### Features with significant changes along each principal component of the latent spaces
plot_nmco_centerPCs_percentiles_chromatin_allfeatures_sampling_groupNMCOde.ipynb - identify chromatin features with significant changes along PCs of the latent spaces  
plot_nmco_centerPCs_percentiles_chromatin_allfeatures_sampling_groupNMCO.ipynb - plot the significant chromatin features  
plot_nmco_centerPCs_percentiles_protein_allfeatures_sampling.ipynb - identify protein features with significant changes along PCs of the latent spaces  
plot_nmco_centerPCs_percentiles_protein_allfeatures_sampling_groupNMCO.ipynb - plot the significant protein features  
#### Feature ablation test of using manually selected features to classify phenotypes
train_clf_conditions_nmco_sampling.ipynb - train phenotype classifier using all represeentative morphological features  
train_clf_conditions_nmco_sampling_featureAblation.ipynb - train phenotype classifier with feature ablation  
plot_clf_conditions_nmco_sampling.ipynb - plot results 

### Image preprocessing
preprocess.ipynb

### Benchmarking
benchmarking_inpainting.ipynb - compare to the previous image inpainting method for protein image prediction: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007348

