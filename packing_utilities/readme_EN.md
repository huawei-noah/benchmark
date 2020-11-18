# 3D Bin Packing Demo, Visualization of Packing Results, Feasibility Check

## Introduction

This program is given to assistant the participants of the EMO 2021 Huawei logistics competition. Three functions are provided in this program:

1. 3D bin packing: generate 3D bin packing results given designated box IDs **(i.e., item IDs)**, route and truck. Note that this is a very simple packing algorithm and the result may not be optimal and can only be used as a baseline.

2. Visualization: generate HTML files for 3D visualization by inputting the order files and packing result files.

3. Feasibility: use the order files and HTML files above to check whether all the constraints are satisfied.

- Function 1: provided by python implementation in "./src/main/python/binpacking_demo";
- Function 2 & 3: provided by java implementation in "./src/main/java/feasibilityCheck";

## How to use it

### 3D Bin Packing

Run the code using Python 3.5. We provide the Pack() class for 3D bin packing, which is initialized with the following parameters:

- input_str: input order string(JSON). See more details in appendix 1;
- spu_ids: boxes to be loaded. Default: all boxes in order string;
- truck_code: truck type. Default: the largest truck in order string;
- route: The order in which the trucks visit the platforms **(i.e., pickup point)**. Default: the order of the platform list in the order string;
- output_path: output directory. Default: do not save the result files.

After creating the instance of Pack() Class, execute the run() function, then an output string (JSON format, see more details in appendix 2) will be returned. If the output_path is specified, the output will be saved into a file named by the **estimateCode** in order string.

**Example:**

`pack = Pack(input_str, spu_ids=["box1", "box2", "box3"], truck_code=["truck_code"], route=["platform1", "platform2", "platform3"], output_path="data/outputs/")`

`pack.run()`


### Result Visualization and Feasibility Check

Run MultiCheck.java with the following parameters:

Parameter 1: the directory of input order (one or more JSON files, see more details in appendix 1).

Parameter 2: the directory of the packing result (one or more JSON files, named by the corresponding input order files. See more details in appendix 2).

Parameter 3: the output path of the feasibility check. Default: "./result/checkResult.txt".

Parameter 4: The output path of 3D visualization. Default: "./result/visualization".

**Example：**

Two ways to execute the program:

1) Compile it by Maven and run the jar file.

Run following codes in terminal with jdk 1.8 and maven 3.3.3 under the directory of feasibility_check:
`maven clean install`
`java -jar feasibility_check-1.0-SNAPSHOT-jar-with-dependencies.jar ./data/inputs/ ./data/outputs/ ./result/checkResult.txt`

2) Run it with jdk 1.8 in an IDE, e.g. IntelliJ Idea.

Download and install IntelliJ Idea, then open the directory of feasibility_check as a Maven Project. Then run the java file feasibilityCheck/MultiCheck.java with the following configuration: 
`Program Arguments: ./data/inputs/ ./data/outputs/ ./result/checkResult.txt`


## Dependencies

This program will automatically download some third party dependencies, which can also be downloaded manually by users. 

Note: if you decide to use these third dependencies, please make sure the requirements of their LICENSE are met.

The following third party dependencies are included:

1. Dependencies that could be downloaded automatically according to pom.xml from Maven central repository: 
`fastjson 1.2.73`
`jsoup 1.13.1`

2. Dependencies defined in file ".\base\base.html.template", which can be accessed online:
`three.js 73`
`jquery.js 1.7.2`
`Stats.js 10`
`Tween.js 0.11.0`

Note: if the online access is not available, you need to download them from their official website into local, and modify line 17-21 of file ".\base\base.html.template" to make it link to the path of the dependent file you downloaded.


# Appendix

## Appendix 1: JSON format of order files (a.k.a. input param files)

| key | key | key | value format | description |
| :-----| :----| :----|:---- |:---- |
| estimateCode |  |  | String | Identification of the input files. |
| algorithmBaseParamDto | | | Object| Basic data |
| | platformDtoList | | Array | Platform (i.e., pick-up point) info. |
| | | platformCode | String | Identification of the platform. |
| | | isMustFirst | Boolean | Whether the platform should be first visited (bonded warehouse). |
| | truckTypeDtoList | | Array | Truck info. |
| | | truckTypeId | String | Identification of the truck. |
| | | truckTypeCode | String | Unique truck code. |
| | | truckTypeName | String | Unique truck name. |
| | | length | Float | Truck length (mm). |
| | | width | Float | Truck width (mm). |
| | | height | Float | Truck height (mm). |
| | | maxLoad | Float | Carrying capacity of the truck (kg) |
| | truckTypeMap |  | Object | Map format of truckTypeDtoList, key is truckTypeId |
| | distanceMap	 |  | Object | Key is two platform codes connected by “+”: e.g. “platform01+platform02”; Value is the float value of the distance (m) between them. |
| boxes	|  |  | Array | Boxes (i.e., items) info. |
| | spuBoxId |  | String | Identification of the box. |
| | platformCode |  | String | Code of the platform the box is belonging to. |
| | length | | Float | Box length (mm). |
| | width |  | Float | Box width (mm). |
| | height |  | Float | Box height (mm). |
| | weight |  | Float | Box weight (kg). |


## Appendix 2: JSON format of packing result files (a.k.a. output param files)

| key | key | key | value format | description |
| :-----| :----| :----|:---- |:---- |
| estimateCode | | | String | Identification of the output file. Same as the input file.|
| solutionArray | | | 2D-Array | Array of solutions. Each line is a solution, i.e. an array of trucks. Each element is a truck map.|
| | truckTypeId | | String | Identification of the truck. Same as the input file.|
| | truckTypeCode | |String | Truck code. Same as the input file.|
| | piece | |Integer | Number of boxes packed in this truck.|
| | volume | |Float | Total volume (mm<sup>3</sup>) of boxes packed in this truck.|
| | weight | | Float | Total weight (kg) of the boxes packed in this truck.|
| | innerLength | | Float | Truck length (mm). Same as the input file. |
| | innerWidth | | Float | Truck width (mm). Same as the input file. |
| | innerHeight | | Float | Truck height (mm). Same as the input file. |
| | maxLoad | | Float | Carrying capacity of the truck (kg). Same as the input file. |
| | platformArray | | Array | Route of the truck, a list of platform codes. |
| | spuArray | | Array | Array of boxes packed in this truck. |
| | | spuId | String | Identification of the box. Same as the input file. |
| | | platformCode | String | Code of the platform the box is belonging to. Same as the input file. |
| | | direction | Integer | Direction of the box (100: box length is parallel to the truck length; 200: box length is perpendicular to the truck length). |
| | | x	| Float | Coordinate of the center of the box (mm) in direction of the truck width, where the origin is the center of the truck. |
| | | y | Float | Coordinate of the center of the box (mm) in direction of the truck height, where the origin is the center of the truck. |
| | | z | Float | Coordinate of the center of the box (mm) in direction of the truck length, where the origin is the center of the truck. |
| | | order | Integer | Order of the box being packed. |
| | | length | Float | Box length (mm). Same as the input file. |
| | | width | Float | Box width (mm). Same as the input file. |
| | | height | Float | Box height (mm). Same as the input file. |
| | | weight | Float | Box weight (mm). Same as the input file. |
