# APOLLO

## Phenotype classification using real images, reconstructed images from full latent space, reconstructed images from shared latent space, or protein images predicted from chromatin
plot_Clf_conditions_sampling.ipynb - plot results
train_clf_conditions_c2c_fullrecon_sampling.ipynb - train classifiers using reconstructed chromatin images from the full latent space  
train_clf_conditions_c2c_sharedrecon_sampling.ipynb - train classifiers using reconstructed chromatin images from the shared latent space  

## Interpretation of partially shared latent spaces of paired chromatin and protein images
### Manually selected chromatin and protein morphological features
getNMCO_allFeatures.ipynb - preprocess  
getNMCOgroups.ipymb - group chromatin features by correlation and selecting one representative feature for each group  
getNMCOgroups_protein.ipymb - group protein features by correlation and selecting one representative feature for each group  
### Plotting examples of chromatin or protein images along each PC
plot_examples_centerPCs_percentiles_noHeldOut.ipynb
### Features with significant changes along each principal component of the latent spaces
plot_nmco_centerPCs_percentiles_chromatin_allfeatures_sampling_groupNMCOde.ipynb - identify chromatin features with significant changes along PCs of the latent spaces  
plot_nmco_centerPCs_percentiles_chromatin_allfeatures_sampling_groupNMCO.ipynb - plot the significant chromatin features  
plot_nmco_centerPCs_percentiles_protein_allfeatures_sampling.ipynb - identify protein features with significant changes along PCs of the latent spaces  
plot_nmco_centerPCs_percentiles_protein_allfeatures_sampling_groupNMCO.ipynb - plot the significant protein features  
### Feature ablation test of using manually selected features to classify phenotypes
plot_clf_conditions_nmco_sampling.ipynb - plot results  
### Image preprocessing
preprocess.ipynb

## Benchmarking
benchmarking_inpainting.ipynb - compare to the previous method for protein image prediction: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007348

