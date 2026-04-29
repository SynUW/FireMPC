I am preparing a NeurIPS Dataset Track paper. The main contributions are as follows:

1. Benchmark dataset

   1. A comprehensive Canadian wildfire risk dataset since the 21st century, including full spatial coverage of 40+ wildfire-related driving factors and wildfire detection data.
   2. The paper should therefore include a fundamental introduction to the dataset, as well as:
   3. Quantitative evaluation of dataset challenges, such as class imbalance, distribution consistency between positive and negative samples (i.e., low separability), and spatiotemporal heterogeneity.

2. A novel sampling strategy

   1. Potential directions include active learning, graph-based methods, or incorporating geographic knowledge, with reference to the literature listed below.
   2. The sampling strategy aims to address the following issues:

      1. Class imbalance is the most critical issue. The ratio of negative to positive samples is approximately 1000:1, making sampling necessary. Existing studies typically use a 2:1 ratio, i.e., two negative samples per positive sample.
      2. Representativeness: ensuring coverage across different land cover types, ecological zones, and months.
      3. Reliability: the MOD14A1 product provides quality control. We select only high-confidence detections and convert the data into binary or ternary labels (e.g., medium, high, and non-fire). However, MODIS products inevitably miss small or short-duration fires and are heavily affected by cloud cover. Under these limitations, how should positive and negative samples be selected?
      4. Challenge: samples should be challenging. What defines “challenging” samples, and how should they be selected? Currently, there is no clear solution.
      5. Experiments should be designed to demonstrate the effectiveness of the proposed sampling strategy.

3. Deep learning baselines

   1. Comparison between spatiotemporal models and purely temporal models.

The overall paper structure and experimental design should be based on these points, with attention to the requirements of a NeurIPS Dataset Track paper.

For the overall structure and experimental setup, refer to dataset papers in remote sensing from NeurIPS in the past three years. Conduct a literature search. If fewer than 10 relevant papers are found, extend the search to other top-tier venues, especially CVPR.

For the sampling strategy, refer to the following papers. Note that only certain aspects may be applicable, so critical reading and evaluation are required. The papers on uncertainty, active learning, sample reliability, and validation analysis are available in Markdown format at:
D:\OneDrive - University of Calgary\个人文档\entire_canada_wildfire_dataset\samplig_references
There are five papers in total. In addition, identify at least five more papers closely related to our task that address the notion of “challenging samples,” preferably from top-tier venues such as Remote Sensing of Environment, CVPR, or NeurIPS.

When designing the sampling strategy, balance implementation simplicity with the level of innovation required for publication. The timeline is approximately 30 days, with around 7 days allocated for data preprocessing (data has already been downloaded). The proposed sampling algorithms must be evaluated along these two dimensions.

Finally, compile the overall paper structure, experimental design and evaluation, and required experiments and analyses into a Markdown document. Save all contents under the directory:
/mnt/raid/user/hackathons/dataset_building_entire_canada/sampling
Include references in the document.
