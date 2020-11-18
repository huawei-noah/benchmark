
# 三维装载demo、装载结果可视化、约束合法性检测

## 功能说明

此脚本包括3个功能：

1. 三维装载：给定箱子ID（即item ID）、车辆路线和车辆类型，进行三维装载（此算法较为简单，不能保证装载效果，仅作为baseline）。

2. 可视化：通过给定格式的输入输出（JSON文件），生成html文件，供三维可视化调试。

3. 合法性检测：在上述HTML文件的基础上，检查解决方案是否满足所有约束。

- 功能1：由src/main/python/binpacking_demo中的python算法实现
- 功能2&3：由src/main/java/feasibilityCheck中的java算法实现

## 使用方法

### 三维装载

在Python 3.5环境中可以运行此代码。我们在pack.py中提供了一个Pack类，用来进行三维装载。输入完整的JSON格式的字符串（入参），并指定要装载的箱子id列表、路径（提货点列表）和车辆代码，函数会输出JSON格式的结果文件（出参）。其初始化函数包括以下参数：

- input_str: 输入订单字符串（JSON格式），详见附录1；
- spu_ids: 待装载箱子的id集合，默认为入参字符串中所有的箱子；
- truck_code: 车辆类型代码，默认为入参字符串中容积最大的车辆；
- route: 车辆访问提货点的顺序，默认为入参字符串中提货点列表的顺序；
- output_path: 输出文件夹路径，输出三维装载结果（JSON文件），默认为不保存；

建立Pack类实例后，运行该实例的run()函数即可返回JSON格式的出参字符串。如果指定了输出文件夹，出参字符串（JSON格式，详见附录2）会保存在此文件夹中，以入参字符串中的estimateCode命名。

**Example：**

`pack = Pack(input_str, spu_ids=["箱子1", "箱子2", "箱子3"], truck_code=["车辆代码"], route=["提货点1", "提货点2", "提货点3"], output_path="data/outputs/")` 

`pack.run()`

### 可视化&约束合法性检测

运行MultiCheck.java文件，在参数中依次输入以下参数，即可运行。

- 参数一：订单文件所在文件夹（一个或多个订单文件）;

- 参数二：订单对应三维装载结果（JSON文件，与订单文件命名相同）所在文件夹;

- 参数三：合法性检查结果输出路径，默认值为./result/checkResult.txt;

- 参数四：可视化文件的输出路径，默认值为./result/visualization文件夹;


此脚本会在算法输出文件夹中生成所有订单的可视化html文件，可使用浏览器（推荐chrome）打开html文件查看三维装载结果，并可拖动放大缩小等查看装载细节。同时，在检查结果文件中，会罗列出所有检查未通过的订单及其未通过的原因。

**Example：**

两种方法：

1）使用Maven进行编译，然后命令行执行jar文件

在安装了jdk 1.8和maven 3.3.3的环境下的命令行运行以下命令：
`maven clean install `
`java -jar feasibility_check-1.0-SNAPSHOT-jar-with-dependencies.jar ./data/inputs/ ./data/outputs/ ./result/checkResult.txt` 

2）在安装了jdk 1.8的环境下用IDE运行程序，如IntelliJ Idea

下载并安装IntelliJ Idea，然后打开feasibility_check文件夹作为一个Maven项目。运行文件feasibilityCheck/MultiCheck.java，运行配置如下： 
`Program Arguments: ./data/inputs/ ./data/outputs/ ./result/checkResult.txt`

## 依赖项

此项目会自动下载若干第三方开源软件依赖，你也可以手动下载。

注意：如果您决定使用这些第三方开源软件，请确保满足其许可证的要求。

依赖的第三方软件如下：

1）可以通过配置pom.xml文件自动从Maven中央库下载的依赖：
`fastjson 1.2.73` 
`jsoup 1.13.1`

1）定义在".\base\base.html.template"文件里的依赖, 自动从网页中获得: 
`three.js 73` 
`jquery.js 1.7.2 `
`Stats.js 10 `
`Tween.js 0.11.0 `

注意：如果线上链接失效，请从这些依赖的官方网站下载到本地，并修改文件".\base\base.html.template"的第17-21行，使其链接到你本地下载的依赖文件路径。


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
