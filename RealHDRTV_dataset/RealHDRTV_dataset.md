# Description
RealHDRTV dataset is the first real-world paired SDRTV-HDRTV dataset, 
which includes SDRTV-HDRTV pairs with 8K resolutions captured by a smartphone camera 
with the “SDR” and “HDR10” modes. To avoid possible misalignment, 
a professional steady tripod is used and only captured indoor or in controlled static scenes. 
After the acquisition, regions are cut out with obvious motions (10+ pixels) and light condition changes, 
and are cropped into 4K image pairs and a global 2D translation is used to align the cropped image pairs. 
Then, the pairs are removed which are still with obvious misalignment and get final 4K SDRTV-HDRTV pairs 
with misalignment no more than 1 pixel as labeled inference dataset.

# Copyright
This dataset is copyright by us and published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. 
This dataset is available for academic use only, and it may not be used for commercial purposes.

# DataSet
The RealHDRTV dataset is provided in Baidu Cloud:
+ <b>Download link:</b> https://pan.baidu.com/s/1qhSNZXyHB_KqKf908hgxMw?pwd=hdr1 
+	<b>Access code:</b> hdr1

# Citation
If using this dataset in your research, please cite the following paper:

    @INPROCEEDINGS{Zhen2022ECCV,    
      author = {Zhen Cheng, Tao Wang, Yong Li, Fenglong Song, Chang Chen and Zhiwei Xiong},      
      title = {Towards Real-World HDRTV Reconstruction: A Data Synthesis-based Approach},      
      booktitle = {European Conference on Computer Vision (ECCV)},      
      year = {2022}      
    }
