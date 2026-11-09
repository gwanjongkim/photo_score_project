# MDIQA Inference Runtime Estimate

## Parameters
- **Target Missing Rows**: 43,582 (Policy B)
- **Observed Speed**: ~44.8 images/second (based on previous 23,102 image inference which completed in 515.0 seconds).
- **Environment**: CUDA visible devices = 0 (GPU inference required).

## Estimates
- **Total Estimated Time**: 971 seconds (~16.2 minutes).
- **Feasibility**: High. The inference job is small enough to be run locally without offloading to a distributed cluster or causing significant blocking time in the pipeline.
