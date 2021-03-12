//  MOEAD_SDVRP 
//  For EMO2021 Huawei VRP competition
//
//  Author:         LIU Fei
//  E-mail:         fliu36-c@my.cityu.edu.hk
//  Create Date:    2021.2.1
//  Last modified   Date: 2021.2.18
//


package jmetal.metaheuristics.moead;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.encodings.variable.Permutation;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.problems.ProblemFactory;
import jmetal.qualityIndicator.QualityIndicator;
import jmetal.util.Configuration;
import jmetal.util.JMException;
import jmetal.problems.CVRP_mix_integer;
import jmetal.util.fast_nondom;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.FileHandler;
import java.util.logging.Logger;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import com.my.vrp.operator.*;
import com.my.vrp.param.L;
import com.my.vrp.utils.PseudoRandom;
import com.my.vrp.Carriage;
import com.my.vrp.Node;
import com.my.vrp.Route;
import com.my.vrp.SolutionSet_vrp;
import com.my.vrp.Solution_vrp;
import com.my.vrp.Solution_vrp;
import com.my.vrp.Box;

/**
 * This class executes the algorithm described in:
 *   H. Li and Q. Zhang, 
 *   "Multiobjective Optimization Problems with Complicated Pareto Sets,  MOEA/D 
 *   and NSGA-II". IEEE Trans on Evolutionary Computation, vol. 12,  no 2,  
 *   pp 284-302, April/2009.  
 */




public class MOEAD_SDVRP {
//  public static Logger      logger_ ;      // Logger object
//  public static FileHandler fileHandler_ ; // FileHandler object
  public static int n_3DPacking ;

  /**
   * @param args Command line arguments. The first (optional) argument specifies 
   *      the problem to solve.
   * @throws JMException 
   * @throws IOException 
   * @throws SecurityException 
   * Usage: three options
   *      - jmetal.metaheuristics.moead.MOEAD_main
   *      - jmetal.metaheuristics.moead.MOEAD_main problemName
   *      - jmetal.metaheuristics.moead.MOEAD_main problemName ParetoFrontFile
   * @throws ClassNotFoundException 
 
   */
  
  
	/**
	 * 从./data/extremes里面读取ideal and nadir points.
	 * @return
	 */
	public static Map<String, Double[]> readExtreme() {
		Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();//problemName->ideal&nadir
		//JSON parser object to parse read file
		JSONParser jsonParser = new JSONParser();
		try (FileReader reader = new FileReader("./data/extremes")){
			JSONObject obj = (JSONObject)jsonParser.parse(reader);//最顶层
			
			JSONArray problemsArray = (JSONArray)obj.get("problems");
			Iterator<JSONObject> iterator = problemsArray.iterator();//用来遍历JSONArray中的JSONObject
			while(iterator.hasNext()) {
				JSONObject curr_problem = iterator.next();
				Double[] idealNadirValues = {(Double)curr_problem.get("idealF1"),(Double)curr_problem.get("idealF2"),(Double)curr_problem.get("nadirF1"),(Double)curr_problem.get("nadirF2")};
				idealNadirMap.put((String)curr_problem.get("problemName"), idealNadirValues);
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (org.json.simple.parser.ParseException e) {
			// TODO Auto-generated catch block 
			e.printStackTrace();
		}
		return idealNadirMap;
	}
  

	public static HashMap input(int fileidx)  throws IOException {
		/**
		 * 初始化ideal和nadir points.
		 */
		HashMap output ;
		output = new HashMap();
		
		//如果极点文件（存两个函数极限小值的文件）存在，则导入数据。
		Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();//problemName->ideal&nadir
		File f = new File("./data/extremes");
		if(f.exists()) {
			idealNadirMap=readExtreme();
		}else {
			//如果不存在，则都设置初始值。
			
			f = new File("./data/inputs");
			String[] filenames = f.list();

			Double [] initialPoints = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MIN_VALUE,Double.MIN_VALUE};
			idealNadirMap.put(filenames[fileidx], initialPoints);

		}
//		System.exit(0);

		
		f = new File("./data/inputs");
		String[] filenames = f.list();
		//统计信息。
//		double total_hv= 0.0;
//		double total_time = 0.0;
//		int total_3dbpp_call = 0;

		
		/**
		 * 选择一个问题。
		 */

		/**
		 * 初始化变量的值。
		 */
//		int n_3dbpp_call = 0;
		long begintime = System.nanoTime();//开始计时
		Double[] idealNadir = (Double[])idealNadirMap.get(filenames[fileidx]);
		int CLIENT_NUM=-1;
		Map<Integer, String> PlatformIDCodeMap =new HashMap<Integer, String>();//platform ID map to code
		Map<String, Integer> PlatformCodeIDMap = new HashMap<String, Integer>();//platform Code map to ID
		int TRUCKTYPE_NUM = -1;//汽车的数量
		ArrayList<Carriage> BASIC_TRUCKS = new ArrayList<Carriage>();
		Map<String, Double> distanceMap = new HashMap<String, Double>();
		
		Node depot_start = new Node();//起始节点
		ArrayList<Node> clients = new ArrayList<Node>();
		ArrayList<Node> clients_trans = new ArrayList<Node>();
		ArrayList<Node> clients_half_trans = new ArrayList<Node>();
		Node depot_end = new Node();//结束节点。
//		int used_truck_id = 0;
		
//		double [][] truck_min_max_lwh;
		double[] truck_min_volume_weight = new double[2];
		truck_min_volume_weight[0] = Double.MAX_VALUE;//记录最小车子的volume
		truck_min_volume_weight[1] = Double.MAX_VALUE;//记录最小车子的weight
//		int truck_min = -1;// 最小车子的下标。
		double[][] client_volume_weight=null;//每个client的所有boxes总的volume和weight
//		System.out.println(filenames[0]);System.exit(0);
//		System.out.println(Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());

		/**
		 * 开始读取输入文件的问题信息。
		 */
		//JSON parser object to parse read file
		JSONParser jsonParser = new JSONParser();
//		String instanceName = "E1594518281316";
		try (FileReader reader = new FileReader("./data/inputs/"+filenames[fileidx])){
			JSONObject obj = (JSONObject)jsonParser.parse(reader);//最顶层
			Iterator<JSONObject> iterator;//用来遍历JSONObject的Array
			JSONObject algorithmBaseParamDto = (JSONObject)obj.get("algorithmBaseParamDto");
			JSONArray platformDtoList = (JSONArray)algorithmBaseParamDto.get("platformDtoList");
//			Iterator<String> iterator = platformDtoList.iterator();
//			while(iterator.hasNext()) {
//				JSONObject currentPlatform = (JSONObject) platformDtoList.get(0);
			CLIENT_NUM = platformDtoList.size();
			boolean [] mustFirst = new boolean [CLIENT_NUM];
			client_volume_weight = new double[CLIENT_NUM][2];
			for(int i=0;i<CLIENT_NUM;i++) {
				client_volume_weight[i][0] = 0.0;
				client_volume_weight[i][1] = 0.0;
			}
			//PlatformMap，文件中platform和代码中platformID的对应关系。
			iterator = platformDtoList.iterator();
			int platformID = 0;
			PlatformIDCodeMap.put(platformID, "start_point");
			PlatformCodeIDMap.put("start_point", platformID);
			platformID++;
			while(iterator.hasNext()) {
				JSONObject platform = iterator.next();
				String platformCode = (String)platform.get("platformCode");
				PlatformIDCodeMap.put(platformID, platformCode);
				PlatformCodeIDMap.put(platformCode, platformID);
				mustFirst[platformID-1] = (boolean)platform.get("mustFirst");
				platformID++;
			}
			PlatformIDCodeMap.put(platformID, "end_point");
			PlatformCodeIDMap.put("end_point", platformID);
			//得到各种类别的卡车
			JSONArray truckTypeDtoList = (JSONArray)algorithmBaseParamDto.get("truckTypeDtoList");
			TRUCKTYPE_NUM = truckTypeDtoList.size();
//			truck_min_max_lwh = new double[3][2];
//			for(int i=0;i<3;i++)
//				truck_min_max_lwh[i][0] = Double.MAX_VALUE;
//			for(int i=0;i<3;i++)
//				truck_min_max_lwh[i][1] = 0;
			for(int basic_truct=0;basic_truct<TRUCKTYPE_NUM;basic_truct++) {
				Carriage truck = new Carriage();
				JSONObject curr_truck = (JSONObject) truckTypeDtoList.get(basic_truct);
				
				truck.setCapacity((double)curr_truck.get("maxLoad"));
				truck.setHeight((double)curr_truck.get("height"));
				truck.setLength((double)curr_truck.get("length"));
				truck.setWidth((double)curr_truck.get("width"));
				truck.setTruckTypeId((String)curr_truck.get("truckTypeId"));
				truck.setTruckTypeCode((String)curr_truck.get("truckTypeCode"));
				BASIC_TRUCKS.add(truck);
				//车子是不是体积越小，载重就越小呢？
				if(truck.getHeight()*truck.getLength()*truck.getWidth()<truck_min_volume_weight[0]) {
					assert(truck.getCapacity()<truck_min_volume_weight[1]);
				}
				if(truck.getHeight()*truck.getLength()*truck.getWidth()<truck_min_volume_weight[0]&&
						truck.getCapacity()<truck_min_volume_weight[1]) {
					truck_min_volume_weight[0] = truck.getHeight()*truck.getLength()*truck.getWidth();
					truck_min_volume_weight[1] = truck.getCapacity();
//					truck_min = basic_truct;
				}

			}
			
			//读取distanceMap
			JSONObject distanceMapJSON = (JSONObject)algorithmBaseParamDto.get("distanceMap");

			//get distance map..
			for(int clienti=1;clienti<=CLIENT_NUM;clienti++) {
				for(int clientj=1;clientj<=CLIENT_NUM;clientj++) {
					if(clienti!=clientj) {
						//不同的client之间的距离从文件里面读取。
						String twoplatforms = PlatformIDCodeMap.get(clienti)+'+'+PlatformIDCodeMap.get(clientj);
//						String twoplatform = clients.get(clienti).getPlatformID());
						distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(clientj), (Double)distanceMapJSON.get(twoplatforms));
					}else {
						//相同的client之间的距离为0.
						distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(clientj), 0.0);
					}
				}
			}
			//从起始点到每个client的距离。
			for(int clienti=1;clienti<=CLIENT_NUM;clienti++) {
				//System.out.println(PlatformIDCodeMap.get(0));
				String twoplatforms = PlatformIDCodeMap.get(0)+'+'+PlatformIDCodeMap.get(clienti);
				distanceMap.put(String.valueOf(0)+'+'+String.valueOf(clienti), (Double)distanceMapJSON.get(twoplatforms));
			}
			//从每个client到终点的距离。
			for(int clienti=1;clienti<=CLIENT_NUM;clienti++) {
				String twoplatforms = PlatformIDCodeMap.get(clienti)+'+'+PlatformIDCodeMap.get(CLIENT_NUM+1);
				distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(CLIENT_NUM+1), (Double)distanceMapJSON.get(twoplatforms));
			}
			
			
			
			//最后读取boxes，初始化start_point, clients, end_point
			
			depot_start.setPlatformID(0);//start no. is always 0
			depot_start.setDemands(0);//start demands is always 0
			depot_start.setGoodsNum(0);//start goodsNum is always 0
			depot_start.setGoods(new ArrayList<Box>());//
			depot_start.setMustFirst(false);
			
			depot_end.setPlatformID(CLIENT_NUM+1);//
			depot_end.setDemands(0);
			depot_end.setGoodsNum(0);
			depot_end.setGoods(new ArrayList<Box>());
			depot_end.setMustFirst(false);
			
			//建立clients所有客户，没有箱子需求。
			for(int i=1;i<=CLIENT_NUM;i++) {
				Node client = new Node();
				ArrayList<Box> boxes = new ArrayList<Box>();
				int platform = i;
				client.setPlatformID(platform);//第几个平台，用于distance-matrix的下标。
				client.setDemands(0);//demands==0,the client's demands are boxes
				client.setGoods(boxes);
				client.setGoodsNum(0);//goods num
				client.setLoadgoodsNum(0);//刚开始所有boxes都没有装载。
				clients.add(client);
				
				Node client_trans = new Node();
				ArrayList<Box> boxes_trans = new ArrayList<Box>();
				int platform_trans = i;
				client_trans.setPlatformID(platform_trans);//第几个平台，用于distance-matrix的下标。
				client_trans.setDemands(0);//demands==0,the client's demands are boxes
				client_trans.setGoods(boxes_trans);
				client_trans.setGoodsNum(0);//goods num
				client_trans.setLoadgoodsNum(0);//刚开始所有boxes都没有装载。
				clients_trans.add(client_trans);
				
				Node client_half_trans = new Node();
				ArrayList<Box> boxes_half_trans = new ArrayList<Box>();
				int platform_half_trans = i;
				client_half_trans.setPlatformID(platform_half_trans);//第几个平台，用于distance-matrix的下标。
				client_half_trans.setDemands(0);//demands==0,the client's demands are boxes
				client_half_trans.setGoods(boxes_half_trans);
				client_half_trans.setGoodsNum(0);//goods num
				client_half_trans.setLoadgoodsNum(0);//刚开始所有boxes都没有装载。
				clients_half_trans.add(client_half_trans);
				
			}
			
			JSONArray boxesJSONArray = (JSONArray)obj.get("boxes");
			iterator = boxesJSONArray.iterator();
			int if_half = 0;
			while(iterator.hasNext()) {
				JSONObject currBoxJSON = iterator.next();
				String platformCode = (String)currBoxJSON.get("platformCode");
				platformID = PlatformCodeIDMap.get(platformCode);
				Box box = new Box();
				box.setSpuBoxID((String)currBoxJSON.get("spuBoxId"));//spuBoxId,specific unique box id
				box.setPlatformid(platformID);
				box.setHeight((double)currBoxJSON.get("height"));//height
				double width = (double)currBoxJSON.get("width");
				double length = (double)currBoxJSON.get("length");
				box.setWidth(width);//width
				box.setLength(length);//length
				box.setWeight((double)currBoxJSON.get("weight"));//fragile or not
				box.setXCoor(0.0);
				box.setYCoor(0.0);
				box.setZCoor(0.0);
				
				Box box_trans = new Box();
				box_trans.setSpuBoxID((String)currBoxJSON.get("spuBoxId"));//spuBoxId,specific unique box id
				box_trans.setPlatformid(platformID);
				box_trans.setHeight((double)currBoxJSON.get("height"));//height
				double width_trans = (double)currBoxJSON.get("length");
				double length_trans = (double)currBoxJSON.get("width");
				if (length > 2318) {
					width_trans = (double)currBoxJSON.get("width");
					length_trans = (double)currBoxJSON.get("length");
				}

				box_trans.setWidth(width_trans);//width
				box_trans.setLength(length_trans);//length
				box_trans.setWeight((double)currBoxJSON.get("weight"));//fragile or not
				box_trans.setXCoor(0.0);
				box_trans.setYCoor(0.0);
				box_trans.setZCoor(0.0);
				
				Box box_half_trans = new Box();
				box_half_trans.setSpuBoxID((String)currBoxJSON.get("spuBoxId"));//spuBoxId,specific unique box id
				box_half_trans.setPlatformid(platformID);
				box_half_trans.setHeight((double)currBoxJSON.get("height"));//height
				double width_half_trans = (double)currBoxJSON.get("length");
				double length_half_trans = (double)currBoxJSON.get("width");
				if (length > 2318 || if_half%2==0) {
					width_half_trans = (double)currBoxJSON.get("width");
					length_half_trans = (double)currBoxJSON.get("length");
				}
				if_half += 1;

				box_half_trans.setWidth(width_half_trans);//width
				box_half_trans.setLength(length_half_trans);//length
				box_half_trans.setWeight((double)currBoxJSON.get("weight"));//fragile or not
				box_half_trans.setXCoor(0.0);
				box_half_trans.setYCoor(0.0);
				box_half_trans.setZCoor(0.0);
//				if(width>length) {
//					box.setDirection(200);
////					System.out.println();
////					System.out.println(filenames[fileidx]);
//				}else {
					box.setDirection(100);
					box_trans.setDirection(100);
					box_half_trans.setDirection(100);
//				}
				//为这个客户添加当前的box
				clients.get(platformID-1).getGoods().add(box);
				clients_trans.get(platformID-1).getGoods().add(box_trans);
				clients_half_trans.get(platformID-1).getGoods().add(box_half_trans);
				//其他信息
				client_volume_weight[platformID-1][0] = client_volume_weight[platformID-1][0] + box.getWidth()*box.getLength()*box.getHeight();
				client_volume_weight[platformID-1][1] = client_volume_weight[platformID-1][1] + box.getWeight();
				
				
			}
			
			//最后对每个客户的boxes进行排序，设置GoodsNum
			for(int i=0;i<CLIENT_NUM;i++) {
//				Collections.sort(clients.get(i).getGoods());//按体积进行从大到小进行排序。
				
				clients.get(i).setGoodsNum(clients.get(i).getGoods().size());//goods num
				clients.get(i).setMustFirst(mustFirst[i]);
				clients_trans.get(i).setGoodsNum(clients_trans.get(i).getGoods().size());//goods num
				clients_trans.get(i).setMustFirst(mustFirst[i]);
				clients_half_trans.get(i).setGoodsNum(clients_half_trans.get(i).getGoods().size());//goods num
				clients_half_trans.get(i).setMustFirst(mustFirst[i]);
			}
//			System.out.println();
//			continue;
			
			
			for(int i=0;i<CLIENT_NUM;i++) {
//				Collections.sort(clients.get(i).getGoods());//按体积进行从大到小进行排序。
				clients.get(i).getGoods().sort(new Comparator<Object>() {
					@Override
					public int compare(Object o1, Object o2) {
						Box b1 = (Box)o1;
						Box b2 = (Box)o2;
						// TODO Auto-generated method stub
						double volume1=b1.getLength()*b1.getWidth();
						double volume2=b2.getLength()*b2.getWidth();
						if(volume1>volume2)
							return -1;
						else if (volume1<volume2)
							return 1;
						else
							if(b1.getHeight()>b2.getHeight())
								return -1;
							else if(b1.getHeight()<b2.getHeight())
								return 1;
							else
								return 0;
					}
					
		
				});
			}
			for(int i=0;i<CLIENT_NUM;i++) {
//				Collections.sort(clients.get(i).getGoods());//按体积进行从大到小进行排序。
				clients_trans.get(i).getGoods().sort(new Comparator<Object>() {
					@Override
					public int compare(Object o1, Object o2) {
						Box b1 = (Box)o1;
						Box b2 = (Box)o2;
						// TODO Auto-generated method stub
						double volume1=b1.getLength()*b1.getWidth();
						double volume2=b2.getLength()*b2.getWidth();
						if(volume1>volume2)
							return -1;
						else if (volume1<volume2)
							return 1;
						else
							if(b1.getHeight()>b2.getHeight())
								return -1;
							else if(b1.getHeight()<b2.getHeight())
								return 1;
							else
								return 0;
					}

					
				});
			}
			for(int i=0;i<CLIENT_NUM;i++) {
//				Collections.sort(clients.get(i).getGoods());//按体积进行从大到小进行排序。
				clients_half_trans.get(i).getGoods().sort(new Comparator<Object>() {

					@Override
					public int compare(Object o1, Object o2) {
						Box b1 = (Box)o1;
						Box b2 = (Box)o2;
						// TODO Auto-generated method stub
						double volume1=b1.getLength()*b1.getWidth();
						double volume2=b2.getLength()*b2.getWidth();
						if(volume1>volume2)
							return -1;
						else if (volume1<volume2)
							return 1;
						else
							if(b1.getHeight()>b2.getHeight())
								return -1;
							else if(b1.getHeight()<b2.getHeight())
								return 1;
							else
								return 0;
					}

					
				});
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (org.json.simple.parser.ParseException e) {
			// TODO Auto-generated catch block 
			e.printStackTrace();
		}
		
		

		output.put("distances",distanceMap);
		output.put("trucks",BASIC_TRUCKS);
		output.put("trucktype_num",TRUCKTYPE_NUM);
		output.put("client_num",CLIENT_NUM);
		output.put("client_v_w",client_volume_weight);
		output.put("clients",clients);
		output.put("clients_trans",clients_trans);
		output.put("clients_half_trans",clients_half_trans);
		output.put("depot_start",depot_start);
		output.put("depot_end",depot_end);
		output.put("PlatformIDCodeMap",PlatformIDCodeMap);

		return output;
	}
  
  
  public static void main(String [] args) throws JMException, SecurityException, IOException, ClassNotFoundException {
//	  String input_directory = args[0];
//		String output_directory = args[1];
	  Problem   problem   ;         // The problem to solve
    Algorithm algorithm ;         // The algorithm to use
    Operator  crossover ;         // Crossover operator
    Operator  mutation  ;         // Mutation operator
//    Operator  crossover_end ;         // Crossover operator
//    Operator  mutation_end  ;         // Mutation operator
    Operator  mutation_mod  ;         // Mutation operator
     
//    QualityIndicator indicators ; // Object to get quality indicators

    HashMap  parameters ; // Operator parameters
    HashMap  results ; 
    HashMap <String, Double> Distances = new HashMap<String, Double>();
    // Logger object and file to store log messages
//    logger_      = Configuration.logger_ ;
//    fileHandler_ = new FileHandler("MOEAD.log"); 
//    logger_.addHandler(fileHandler_) ;
    
	Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();//problemName->ideal&nadir
	File f_ideal = new File("./data/extremes");
	if(f_ideal.exists()) {
		idealNadirMap=readExtreme();
	}else {
		//如果不存在，则都设置初始值。
		
		f_ideal = new File("./data/inputs");
		String[] file_ideal_names = f_ideal.list();
		for(int fileidx=0;fileidx<file_ideal_names.length;fileidx++) {
		Double [] initialPoints = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MIN_VALUE,Double.MIN_VALUE};
		idealNadirMap.put(file_ideal_names[fileidx], initialPoints);
		}
	}
	
	double total_hv= 0.0;
	double total_time = 0.0;
//	int total_3dbpp_call = 0;
	
    
//    indicators = null ;
	File f = new File("./data/inputs");
	String[] filenames = f.list();
    for (int file_id = 0; file_id < filenames.length; file_id++) {
    	
    	n_3DPacking = 0;
    	

    	
    	long begintime = System.nanoTime();
    	
		HashMap Data = new HashMap();
		ArrayList<Carriage> Trucks = new ArrayList<Carriage>();
	    Data = input(file_id);
	    Distances  = (HashMap<String, Double>) Data.get("distances");
	    Trucks  = (ArrayList<Carriage>) Data.get("trucks");
		int Trucktype_num = (int) Data.get("trucktype_num");
		int NUM_nodes = (int) Data.get("client_num");
		ArrayList<Node> clients =  (ArrayList<Node>) Data.get("clients");
		ArrayList<Node> clients_trans =  (ArrayList<Node>) Data.get("clients_trans");
		ArrayList<Node> clients_half_trans =  (ArrayList<Node>) Data.get("clients_half_trans");

		HashMap<Integer, String> PlatformIDCodeMap = (HashMap<Integer, String>) Data.get("PlatformIDCodeMap");
		double[] VEHICLE_CAPACITY = new double[Trucktype_num];//每輛車的載重。
		double[] VEHICLE_VOLUME = new double[Trucktype_num];//每輛車的體積。
		
	    for (int i=0; i<Trucktype_num; i++ ) {
	    	VEHICLE_CAPACITY[i] = (double) Trucks.get(i).getCapacity();
	    	//System.out.println(Trucks.get(i).getWidth());
	    	//System.out.println(Trucks.get(i).getLength());
	    	//System.out.println(Trucks.get(i).getHeight());
	    	//System.out.println(Trucks.get(i).getTruckVolume());
	    	VEHICLE_VOLUME[i] = (double) Trucks.get(i).getTruckVolume();
	    }
	    //double[][] client_v_w = new double[NUM_nodes][2];
	    double[][] client_v_w = (double[][]) Data.get("client_v_w"); 
	    
	    int[] if_large_box = new int[clients.size()];
	    double[] if_hard_node = new double[clients.size()];
	    double[] if_hard_node0 = new double[clients.size()];
	    ArrayList<Double> v_different = new ArrayList<Double>();//有哪些不同體積的箱子。

	    
		for (int i=0; i<clients.size(); i++) {
			//對於節點i
			if_hard_node[i] = 1.0;
			if_hard_node0[i] = 1.0;
			double max_v_box = 0;
			int max_v_box_n = 0;
		    double[] vs = new double[10000];
		    int[] vs_no = new int[10000];

			for (int j=0; j<clients.get(i).getGoodsNum(); j++) {
				if (clients.get(i).getGoods().get(j).getHeight() > Trucks.get(0).getHeight()){
					if_large_box[i] = 1;
					break;
				}
			}
			//System.out.println(clients.get(i).getDemands());
			double node_v_all = 0;//所有箱子的體積
			for (int nbox=0; nbox<clients.get(i).getGoodsNum();nbox++) {
				node_v_all += clients.get(i).getGoods().get(nbox).getVolume();
			}
			if(node_v_all>0.15*VEHICLE_VOLUME[Trucktype_num-1]) {
				
				for (int j=0; j<clients.get(i).getGoodsNum(); j++) {
					int if_add=1;
					double current_v = clients.get(i).getGoods().get(j).getVolume();
					if (current_v >= max_v_box){//哪個箱子的體積最大。
						max_v_box_n = j;
						max_v_box = current_v;
					}
					if (j==0) {
						v_different.add(current_v);
						vs[0] += current_v;//
						vs_no[0] = j;
					}
					else {
						//每種箱子的體積。
						for (int vd=0; vd<v_different.size();vd++) {
							if (current_v == v_different.get(vd)) {
								vs[vd] += current_v;
								vs_no[vd] = j;
								if_add = 0;//表示有相同體積的箱子已經加了。
								break;						
							}
						}
						if(if_add==1) {//如果這個箱子沒有相同體積的箱子，則增加。
							v_different.add(current_v);
						}
					}			
				}
				
				int max_vs_box_n = 0;//
				double max_vs = 0;//有相同的箱子的體積最大
				for (int vd=0; vd<v_different.size();vd++) {
					if(vs[vd]>max_vs) {
						max_vs = vs[vd];
						max_vs_box_n = vs_no[vd];
					}
				}

				//最大的當個箱子。
				Box max_box = clients.get(i).getGoods().get(max_v_box_n);
				//System.out.println(max_box.getHeight());
				//System.out.println(max_box.getWidth());
				if (max_box.getHeight()>0.5*Trucks.get(Trucktype_num-1).getHeight()&& max_box.getHeight()<0.75*Trucks.get(Trucktype_num-1).getHeight()) {
					if_hard_node[i] = 0.8;//體積最大的箱子
				}
				if (max_box.getWidth()>0.5*Trucks.get(Trucktype_num-1).getWidth()&& max_box.getWidth()<0.75*Trucks.get(Trucktype_num-1).getWidth()) {
					if_hard_node[i] = 0.8;
				}
				if (max_box.getHeight()>0.33334*Trucks.get(Trucktype_num-1).getHeight()&& max_box.getHeight()<0.4*Trucks.get(Trucktype_num-1).getHeight()) {
					if_hard_node[i] = 0.85;
				}
				if (max_box.getWidth()>0.333334*Trucks.get(Trucktype_num-1).getWidth()&& max_box.getWidth()<0.4*Trucks.get(Trucktype_num-1).getWidth()) {
					if_hard_node[i] = 0.85;
				}
				
				//有最大體積的箱子。
				Box max_boxs = clients.get(i).getGoods().get(max_vs_box_n);
				//System.out.println(max_boxs.getHeight());
				//System.out.println(max_boxs.getWidth());
				if (clients.get(i).getGoods().get(max_vs_box_n).getHeight()!=clients.get(i).getGoods().get(max_v_box_n).getHeight()) {
					max_boxs = clients.get(i).getGoods().get(max_vs_box_n);
					//System.out.println(max_boxs.getHeight());
					//System.out.println(max_boxs.getWidth());
					if (max_boxs.getHeight()>0.5*Trucks.get(Trucktype_num-1).getHeight()&& max_boxs.getHeight()<0.75*Trucks.get(Trucktype_num-1).getHeight()) {
						if_hard_node[i] = 0.8;
					}
					if (max_boxs.getWidth()>0.5*Trucks.get(Trucktype_num-1).getWidth()&& max_boxs.getWidth()<0.75*Trucks.get(Trucktype_num-1).getWidth()) {
						if_hard_node[i] = 0.8;
					}
					if (max_boxs.getHeight()>0.33334*Trucks.get(Trucktype_num-1).getHeight()&& max_boxs.getHeight()<0.4*Trucks.get(Trucktype_num-1).getHeight()) {
						if_hard_node[i] = 0.85;
					}
					if (max_boxs.getWidth()>0.333334*Trucks.get(Trucktype_num-1).getWidth()&& max_boxs.getWidth()<0.4*Trucks.get(Trucktype_num-1).getWidth()) {
						if_hard_node[i] = 0.85;
					}
				}
			}
		}
		
		
	    /**
	     * 找到每個節點的裝載率。
	     * truck_weight_ratio[i]==1.0表示這個節點用一輛車裝夠了。
	     */
	    double[] truck_weight_ratio = new double[NUM_nodes];  
	    for (int i = 0; i < NUM_nodes; i++) {
	    	truck_weight_ratio[i] = 1.0;
	    	if (client_v_w[i][0]>1.1*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {//這個節點箱子的體積比較大。很大概率一輛車裝不完。
	    		for (double nr=1.0 ;nr>0.549; nr-=0.02) {
	    			truck_weight_ratio[i] = nr;
	    			//為node i 建立一條路徑。
		    		Route route = new Route((int)(1000+Math.random()*8999));
		    		
		        	ArrayList<Box> unloadboxes = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_trans = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_half_trans = new ArrayList<Box>();
		        	
		        	double volume_precent = 0.0;
		        	int ngoods = clients.get(i).getGoodsNum();
		        	//ArrayList<Box> goods = new ArrayList<Box>();
		        	for (int k = 0; k < ngoods; k++) {
		        		volume_precent += clients.get(i).getGoods().get(k).getVolume();
		        		if( volume_precent >= truck_weight_ratio[i]*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
		        			break;
		        		}
		        		//goods.add(clients.get(node_id).getGoods().get(k));
		        		unloadboxes.add(clients.get(i).getGoods().get(k));
		        		unloadboxes_trans.add(clients_trans.get(i).getGoods().get(k));
		        		unloadboxes_half_trans.add(clients_half_trans.get(i).getGoods().get(k));
		        		//client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
		        	}
		        	
		        	ArrayList<Integer> k = null;

						Carriage vehicle1 = new Carriage(Trucks.get(VEHICLE_CAPACITY.length-1));//用最後一種車型。
						route.setCarriage(vehicle1);	
	
							int if_pack = 1;
							double unload_v = 0;//有多少體積沒有裝載的。
							for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
								if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {//這個箱子不能用這個車子裝。
									if_pack = 0;
								}
								if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
									if_pack = 0;
								}
								if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
									if_pack = 0;
								}
								unload_v += unloadboxes.get(nbox).getVolume();
							}
							if (unload_v > vehicle1.getTruckVolume()) {//如果要裝載的箱子的體積更大，也不用pack了。
								if_pack = 0;
							}
							if (if_pack == 1) {
								k = route.zqlbpp(unloadboxes,2);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
								k = route.dblf(unloadboxes,1);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
								k = route.zqlbpp(unloadboxes_trans,2);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
								k = route.dblf(unloadboxes_trans,1);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
								k = route.zqlbpp(unloadboxes_half_trans,2);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
								k = route.dblf(unloadboxes_half_trans,1);
								//n_3DPacking += 1;
								if (k.size() == unloadboxes.size()) {
									break;
								}
								//System.out.println(k.size());
							}
							n_3DPacking += 1;
	    		}
	    	}//if 
	    }//for each node
	    		

	    
	    // main settings *************************************************
	    int n_repeat = 25;
	    int n_split = 4; 
	    double[] relax_ratio_all = {0.95,0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55,0.525,0.5,0.475,0.45,0.425,0.4,0.375,0.35};
	    //double[] relax_ratio_all = {0.35,0.38,0.4,0.45,0.475,0.5,0.55,0.575,0.6,0.625,0.65,0.7,0.75,0.8,0.85};
	    // main settings finished *************************************************
	    
	    //double[][] save_coords = new double[n_repeat*n_split][2] ;
	    HashMap PF_out = new HashMap();
	    
	    //int[] VEHICLE_CAPACITY = {50,60,60};
	    //int NUM_nodes = 10;
	    
	    //ArrayList <double[]> save_all_tested_objs = new ArrayList <double[]>();
    	ArrayList<double[]> best_results = new ArrayList <double[]>();
	    
//	    SolutionSet population_last = null;
	    SolutionSet_vrp solutionSet_last = new SolutionSet_vrp();
	    for(int runs = 0; runs < n_repeat; runs++) {
	    	double [] if_hard_node_use = if_hard_node0;
	    	if (runs%3==0) {
	    		if_hard_node_use = if_hard_node;
	    	}
	    	for (int relax_r = 0; relax_r < relax_ratio_all.length; relax_r++) {//
	    	
	    	double relax_ratio = relax_ratio_all[relax_r];

		    
		    for (int split_w = 1; split_w < n_split+1; split_w++) {//1,2,3,4
		    	
		    		
			    ArrayList<double[]> save_objs = new ArrayList<double[]>();
			    ArrayList<Double> save_consts = new ArrayList<Double>();
			    ArrayList<Solution> save_vars = new ArrayList<Solution>() ; 

			   // ArrayList<double[]> save_objs0 = new ArrayList<double[]>();
		    		
			    //double[] all_split_minv = new double[100000]; 
			    int n_solution = 0;
			    //int n_all = 0;
			    
			    double split_minv = 0.0 ;//超過多少就開始分割。
			    if (split_w==1) {
			    	split_minv = 0.3*VEHICLE_VOLUME[0];
			    }
			    else if (split_w==2){
			    	split_minv = VEHICLE_VOLUME[0];
			    }
			    else if (split_w==3){
			    	split_minv = 0.5*VEHICLE_VOLUME[VEHICLE_VOLUME.length-1];
			    }
			    else if (split_w==4){
			    	split_minv = 0.8*VEHICLE_VOLUME[VEHICLE_VOLUME.length-1];
			    }
		    	//double split_minv =   ((double) split_w/ (double) n_split)*(max_v-min_v)+min_v-100.0; 
		    	
		    	problem = new CVRP_mix_integer("PermutationBinary",split_minv,VEHICLE_CAPACITY,VEHICLE_VOLUME,NUM_nodes,Distances,client_v_w,relax_ratio,truck_weight_ratio,clients,if_large_box,if_hard_node_use);
		
		    	algorithm = new cMOEAD(problem);
		
		        // Algorithm parameters
		        algorithm.setInputParameter("populationSize",80);
		        algorithm.setInputParameter("maxEvaluations",6000);
		        
		
		        algorithm.setInputParameter("dataDirectory",
		        "/Users/antelverde/Softw/pruebas/data/MOEAD_parameters/Weight");
		
//		        algorithm.setInputParameter("finalSize", 60) ; // used by MOEAD_DRA
		
		        algorithm.setInputParameter("T", 25) ;
		        algorithm.setInputParameter("delta", 0.9) ;
		        algorithm.setInputParameter("nr", 2) ;
		
		        /* Crossver operator */
		        parameters = new HashMap() ;
		        parameters.put("PMXCrossoverProbability", 0.95) ;
		        parameters.put("binaryCrossoverProbability", 0.9) ;
		        //crossover = CrossoverFactory.getCrossoverOperator("TwoPointsCrossover", parameters);
		        crossover = CrossoverFactory.getCrossoverOperator("PMXsinglepointCrossover", parameters);                
		        
		        /* Mutation operator */
		        parameters = new HashMap() ;
		        parameters.put("SwapMutationProbability", 0.12) ;
		        parameters.put("binaryMutationProbability", 0.12) ;
		        mutation = MutationFactory.getMutationOperator("SwapBitFlipMutation", parameters);  
		        
	        
		        /* Mutation operator */
		        parameters = new HashMap() ;
		        parameters.put("SwapMutationProbability", 0.5) ;
		        //parameters.put("binaryMutationProbability", 0.12) ;
		        mutation_mod = MutationFactory.getMutationOperator("SwapMutation_mod", parameters);  
		        
		        
		        algorithm.addOperator("crossover",crossover);
		        algorithm.addOperator("mutation",mutation);
		        
//		        algorithm.addOperator("crossover_end",crossover_end);
//		        algorithm.addOperator("mutation_end",mutation_end);
		        algorithm.addOperator("mutation_mod",mutation_mod);
		        
		        // Execute the Algorithm
//		        long initTime = System.currentTimeMillis();
		        SolutionSet population = algorithm.execute();
//		        population_last = population;
//		        long estimatedTime = System.currentTimeMillis() - initTime;

		        results = population.getresults();
		        
		        final List<Solution> solutions;
		        solutions = (List<Solution>) results.get("result");
		        
		        for (Solution aSolutionsList_ : solutions) {
		        	save_objs.add(aSolutionsList_.objective_);
		        	save_vars.add(aSolutionsList_);
		        	save_consts.add(aSolutionsList_.overallConstraintViolation_);
		            //n_solution += 1;
		        	//all_split_minv [n_all] = split_minv;
		            //n_all +=1;
		          }
		
		        ArrayList<double[]> save_objs0 = new ArrayList<double[]>();
			    ArrayList<Solution> save_vars0 = new ArrayList<Solution>() ;
			    
		        for (int n = 0; n<save_objs.size(); n++) {
		        	boolean if_add = true;
		        	if (save_consts.get(n) < 0) {//不要不可行解
		        		if_add = false;continue;
		        	}
		        	for (int nn=0; nn<best_results.size(); nn++ ) {
		        		//如果被當前最好的解支配
		        		if (save_objs.get(n)[0]>best_results.get(nn)[0]&&save_objs.get(n)[1]>best_results.get(nn)[1]) {
		        			if_add = false;break;
		        		}
		        	}
		        	if(if_add) {
			        	save_objs0.add(save_objs.get(n));
			        	save_vars0.add(save_vars.get(n));
		        	}
		        }
		        
		        if (save_objs0.size()>0) {//如果這次運行，有解。
		        	
			        ArrayList<double[]> save_objs1 = new ArrayList<double[]>();
				    ArrayList<Solution> save_vars1 = new ArrayList<Solution>() ;

		        				    
		        	fast_nondom FND = new fast_nondom();
		        	
			        PF_out = FND.fast_nondominated_sort(save_objs0);
			        n_solution = (int)PF_out.get("n");
//			        double[][] PF = (double[][]) PF_out.get("PF");
//			        int[] no_PF =(int[])PF_out.get("no_PF");

				    int[] save_feasible_no = new int[n_solution];
				    
				    for (int n = 0; n<n_solution; n++) {
				    		save_vars1.add(save_vars0.get(((int[])PF_out.get("no_PF"))[n]));
				    		save_objs1.add(save_objs0.get(((int[])PF_out.get("no_PF"))[n]));		        
				    }
				    
				    SolutionSet_vrp solutionSet = get_Solutions(split_minv,save_vars1,file_id,relax_ratio,truck_weight_ratio,save_feasible_no,if_hard_node_use);
				    
				    for (int n_solu=0;n_solu<solutionSet.size();n_solu++) {
					    solutionSet_last.add(solutionSet.get(n_solu));
				    }
				    for (int n = 0; n<n_solution; n++) {
				    	if(save_feasible_no[n]==1) {
				    		best_results.add(save_objs0.get(n));				    		
				    	}
				    }
				    
		        }//if save_objs>0
		    	}//for split_w
		        if(n_3DPacking>=2000 || (n_3DPacking >= 1950 && solutionSet_last.size()==0)) {
		        	break;
		        }
		    }//for relax_r
		    //System.out.println(runs);
	        if(n_3DPacking>=2000 || (n_3DPacking >= 1950 && solutionSet_last.size()==0)) {
	        	break;
	        }
	    }
	    
        if(solutionSet_last.size()==0) {
        	SolutionSet_vrp solutionSet = get_Solutions_naive(file_id,truck_weight_ratio);
		    for (int n_solu=0;n_solu<solutionSet.size();n_solu++) {
			    solutionSet_last.add(solutionSet.get(n_solu));
		    }
        }

	    
	    if (solutionSet_last.size()>0) {
			solutionSet_last.removeDomintes();
			
			for (int i=0;i<solutionSet_last.size();i++) {
//				System.out.println("objs: "+solutionSet_last.get(i).getF1());
//				System.out.println("objs: "+solutionSet_last.get(i).getF2());
			}
	    	
			//System.out.print(solutionSet.solutionList_.size()+"\t");
			/**************统计当前问题的结果**************/
			long endtime = System.nanoTime();
			double usedTime= (endtime - begintime)/(1e9);
			

			Double[] idealNadir = (Double[])idealNadirMap.get(filenames[file_id]);
			double hv = solutionSet_last.get2DHV(idealNadir);
			System.out.print("File: "+filenames[file_id]+"\t");
			System.out.printf("estimated hv: %6.5f \t",hv);
			System.out.printf("time: %6.5f s\t",usedTime);
			System.out.print("3dbpp no: "+	n_3DPacking );
			//System.out.print("objs: "+solutionSet.get(0).getF1());
			//System.out.print("objs: "+solutionSet.get(0).getF2());
			total_hv = total_hv + hv;
			total_time = total_time+usedTime;
//			total_3dbpp_call = total_3dbpp_call+n_3DPacking;
			//System.out.println();
			//if(isUpdateExtreme)
			//idealNadirMap.put(filenames[fileidx], idealNadir);//更新ideal 和nadir point
			System.out.println();
			outputJSON(solutionSet_last, filenames[file_id], PlatformIDCodeMap, "./data/outputs/"+filenames[file_id]);
			for(int solutioni=0;solutioni<solutionSet_last.size();solutioni++)
				outputJSON(solutionSet_last.get(solutioni), filenames[file_id], PlatformIDCodeMap, "./data/outputs_test/finalsolution");
			
//		    for (int n = 0; n<n_solution; n++) {
//				problem = new CVRP_mix_integer("PermutationBinary",all_split_minv[((int[])PF_out.get("no_PF"))[n]],VEHICLE_CAPACITY,VEHICLE_VOLUME,NUM_nodes,Distances,client_v_w,relax_ratio);
//			    problem.evaluate_check(save_vars0.get(n));
//			    problem.evaluateConstraints_final (save_vars0.get(n));
//			    //System.out.println(save_vars0.get(n).overallConstraintViolation_);
//		    }
		    //System.out.println("check");	    	
	    }
	    else {
	    	System.out.println("can not find a solution on "+filenames[file_id]+" !!!");
	    	
		    FileOutputStream fos   = new FileOutputStream(filenames[file_id]+"_error.dat")     ;
		    OutputStreamWriter osw = new OutputStreamWriter(fos)    ;
		    BufferedWriter bw      = new BufferedWriter(osw)        ;   
			bw.newLine();
	        bw.close();

	    }


    }
	System.out.println("Total estimated HV:\t"+total_hv);
	System.out.println("Total Time:\t"+total_time);
//	System.out.println("Total 3dbpp call:\t"+total_3dbpp_call);
    
    //logger_.info("Path values have been writen to file Path");
    //population_last.printFinalPathToFile("FinalPath",all_split_minv,save_vars0,(int[])PF_out.get("no_PF"),VEHICLE_CAPACITY,NUM_nodes);

     
    
        
  } //main
  
	static void outputJSON(Solution_vrp solution, String instanceName, Map<Integer, String> PlatformIDCodeMap, String outputFile) {
		//output to json file.
				JSONObject outputJSONObject = new JSONObject();
				//{
				//"estimateCode":"E1594518281316",
				outputJSONObject.put("estimateCode",instanceName);
				//***************************************************准备truckArray
				JSONArray truckArray = new JSONArray();
				
//				Iterator<Route> iteratorRoute = solution.getRoutes().iterator();
//				while(iteratorRoute.hasNext()) {
				for(int routei=0;routei<solution.getRoutes().size();routei++) {
//				Carriage currTruck = .getCarriage();
				Route route = solution.getRoutes().get(routei);
				//一辆车
				JSONObject truckJSONObject = new JSONObject();
				//这辆车基本信息 1. innerHeight
				truckJSONObject.put("innerHeight", route.getCarriage().getHeight());
				
				//这辆车经过的路径信息->2. platformArray
				ArrayList<String> platformArray = new ArrayList<String>();
				for(int nodei=1;nodei<route.getNodes().size()-1;nodei++) {
					platformArray.add(PlatformIDCodeMap.get(route.getNodes().get(nodei).getPlatformID()));
				}
//				
//				while(iterator.hasNext())
//					if(iterator.)
//					
				truckJSONObject.put("platformArray", platformArray);
				truckJSONObject.put("volume", route.getLoadVolume());
				truckJSONObject.put("innerWidth", route.getCarriage().getWidth());
				truckJSONObject.put("truckTypeCode", route.getCarriage().getTruckTypeCode());
				truckJSONObject.put("piece", route.getBoxes().size());//number of boxes
				JSONArray spuArray = new JSONArray();//the boxes array
				
//				Iterator<Box> iteratorBox = route.getCarriage().getBoxes().iterator();
				int order = 1;
//				while(iteratorBox.hasNext()) {
				for(int boxi=0;boxi<route.getBoxes().size();boxi++) {
					Box box = route.getBoxes().get(boxi);//iteratorBox.next();
					JSONObject currBox = new JSONObject();// the current box information
					currBox.put("spuId", box.getSpuBoxID());
					currBox.put("order", order);order=order+1;
					currBox.put("direction", box.getDirection());//length parallel to the vehicle's length
					currBox.put("x", box.getXCoor()+box.getWidth()/2.0-route.getCarriage().getWidth()/2.0);//-box.getWidth()
					currBox.put("y", box.getYCoor()+box.getHeight()/2-route.getCarriage().getHeight()/2.0);//
					currBox.put("length", box.getLength());
					currBox.put("weight", box.getWeight());
					currBox.put("height", box.getHeight());
					currBox.put("width", box.getWidth());
					currBox.put("platformCode", PlatformIDCodeMap.get(box.getPlatformid()));
					currBox.put("z", box.getZCoor()+box.getLength()/2-route.getCarriage().getLength()/2.0);//-box.getHeight()
					spuArray.add(currBox);
				}
				truckJSONObject.put("spuArray", spuArray);//the array of boxes
				truckJSONObject.put("truckTypeId", route.getCarriage().getTruckTypeId());
				truckJSONObject.put("innerLength", route.getCarriage().getLength());
				truckJSONObject.put("maxLoad", route.getCarriage().getCapacity());
				truckJSONObject.put("weight", route.getLoadWeight());
				truckArray.add(truckJSONObject);
				}
				//***************************************************
				outputJSONObject.put("truckArray",truckArray);
				
				File f = new File(outputFile);
				if(!f.getParentFile().exists()) {
					f.getParentFile().mkdir();
				}
//				"./data/outputs"+used_truck_id+"/"+instanceName
				try (FileWriter file = new FileWriter(outputFile)) {
		            file.write(outputJSONObject.toJSONString());
		        } catch (IOException e) {
		            e.printStackTrace();
		        }
	}
	
	
	
	static void outputJSON(SolutionSet_vrp solutionSet, String instanceName, Map<Integer, String> PlatformIDCodeMap, String outputFile) {
		//output to json file.
		JSONObject outputJSONObject = new JSONObject();
		//{
		//"estimateCode":"E1594518281316",
		outputJSONObject.put("estimateCode",instanceName);
		
		ArrayList<JSONArray> solutionArray = new ArrayList<JSONArray>();
		for(int solutioni=0;solutioni<solutionSet.size();solutioni++) {
			Solution_vrp solution = solutionSet.get(solutioni);
//				Iterator<Route> iteratorRoute = solution.getRoutes().iterator();
//				while(iteratorRoute.hasNext()) {
			//***************************************************准备truckArray
			JSONArray truckArray = new JSONArray();
			for(int routei=0;routei<solution.getRoutes().size();routei++) {
//				Carriage currTruck = .getCarriage();
				Route route = solution.getRoutes().get(routei);
				//一辆车
				JSONObject truckJSONObject = new JSONObject();
				//这辆车基本信息 1. innerHeight
				truckJSONObject.put("innerHeight", route.getCarriage().getHeight());
				
				//这辆车经过的路径信息->2. platformArray
				ArrayList<String> platformArray = new ArrayList<String>();
				for(int nodei=1;nodei<route.getNodes().size()-1;nodei++) {
					platformArray.add(PlatformIDCodeMap.get(route.getNodes().get(nodei).getPlatformID()));
				}
//				
//				while(iterator.hasNext())
//					if(iterator.)
//					
				truckJSONObject.put("platformArray", platformArray);
				truckJSONObject.put("volume", route.getLoadVolume());
				truckJSONObject.put("innerWidth", route.getCarriage().getWidth());
				truckJSONObject.put("truckTypeCode", route.getCarriage().getTruckTypeCode());
				truckJSONObject.put("piece", route.getBoxes().size());//number of boxes
				JSONArray spuArray = new JSONArray();//the boxes array
				
//				Iterator<Box> iteratorBox = route.getCarriage().getBoxes().iterator();
				int order = 1;
//				while(iteratorBox.hasNext()) {
				for(int boxi=0;boxi<route.getBoxes().size();boxi++) {
					Box box = route.getBoxes().get(boxi);//iteratorBox.next();
					JSONObject currBox = new JSONObject();// the current box information
					currBox.put("spuId", box.getSpuBoxID());
					currBox.put("order", order);order=order+1;
					currBox.put("direction", box.getDirection());//length parallel to the vehicle's length
					currBox.put("x", box.getXCoor()+box.getWidth()/2.0-route.getCarriage().getWidth()/2.0);//-box.getWidth()
					currBox.put("y", box.getYCoor()+box.getHeight()/2-route.getCarriage().getHeight()/2.0);//
					currBox.put("length", box.getLength());
					currBox.put("weight", box.getWeight());
					currBox.put("height", box.getHeight());
					currBox.put("width", box.getWidth());
					currBox.put("platformCode", PlatformIDCodeMap.get(box.getPlatformid()));
					currBox.put("z", box.getZCoor()+box.getLength()/2-route.getCarriage().getLength()/2.0);//-box.getHeight()
					spuArray.add(currBox);
				}
				truckJSONObject.put("spuArray", spuArray);//the array of boxes
				truckJSONObject.put("truckTypeId", route.getCarriage().getTruckTypeId());
				truckJSONObject.put("innerLength", route.getCarriage().getLength());
				truckJSONObject.put("maxLoad", route.getCarriage().getCapacity());
				truckJSONObject.put("weight", route.getLoadWeight());
				truckArray.add(truckJSONObject);
			}//routei
			solutionArray.add(truckArray);
		}//solutioni
		//***************************************************
		outputJSONObject.put("solutionArray",solutionArray);
		
		File f = new File(outputFile);
		if(!f.getParentFile().exists()) {
			f.getParentFile().mkdir();
		}
//				"./data/outputs"+used_truck_id+"/"+instanceName
		try (FileWriter file = new FileWriter(outputFile)) {
            file.write(outputJSONObject.toJSONString());
        } catch (IOException e) {
            e.printStackTrace();
        }
	}
	
//  get_Solutions 
//
//  For EMO2021 Huawei VRP competition
//
//
	
	static SolutionSet_vrp get_Solutions(double  split_minv, ArrayList<Solution> solutionsList_0,int file_id, double relax_ratio,double[] truck_weight_ratio, int[] save_feasible_no, double[]if_hard_node) throws IOException {

		HashMap <String, Double> Distances = new HashMap<String, Double>();		
		HashMap Data = new HashMap();
		ArrayList<Carriage> BASIC_TRUCKS = new ArrayList<Carriage>();
		Data = input(file_id);
		Distances  = (HashMap<String, Double>) Data.get("distances");
		BASIC_TRUCKS  = (ArrayList<Carriage>) Data.get("trucks");
		int Trucktype_num = (int) Data.get("trucktype_num");
		int NUM_nodes = (int) Data.get("client_num");
		double[] VEHICLE_CAPACITY = new double[Trucktype_num];
		double[] VEHICLE_VOLUME = new double[Trucktype_num];
		
		for (int i=0; i<Trucktype_num; i++ ) {
			VEHICLE_CAPACITY[i] = (double) BASIC_TRUCKS.get(i).getCapacity();
			VEHICLE_VOLUME[i] = (double) BASIC_TRUCKS.get(i).getTruckVolume();
		}

		double[][] client_v_w = (double[][]) Data.get("client_v_w");
		ArrayList<Node> clients =  (ArrayList<Node>) Data.get("clients");
		ArrayList<Node> clients_trans =  (ArrayList<Node>) Data.get("clients_trans");
		ArrayList<Node> clients_half_trans =  (ArrayList<Node>) Data.get("clients_half_trans");
		Node depot_start = (Node) Data.get("depot_start");
		Node depot_end = (Node) Data.get("depot_end");
		
		//double relax_volume = relax_ratio*VEHICLE_VOLUME[0];
	  
	  
	    int Type_trucks = VEHICLE_CAPACITY.length;
	    double Split_precision = 6.0;
	    int no_results = 0;
	    int x = 0 ;
	    //int numberOfVariables = solutionsList_0.get(0).getDecisionVariables().length ;
	    //int n_solutions = 0;
	    
   
	    SolutionSet_vrp solutionSet = new SolutionSet_vrp();

	  
	    for (int n_sol =0; n_sol <solutionsList_0.size(); n_sol++) {
	    	
	    	Solution aSolutionsList_ = solutionsList_0.get(n_sol);
	    			
	        if(n_3DPacking>=2000) {
	        	break;
	        }
	    	
		    Solution_vrp solution = new Solution_vrp();
		    solution.distanceMap=Distances;
	      	
	        int n_loadboxes = 0;
	        int n_loadboxes_check = 0;
		    int new_num_nodes = 0;
		    int numberOfCities_ = 0;
		    
		    int[] nodes_record = new int[NUM_nodes];
		    
//		    //Box unloadboxes0 = new Box();
//		    ArrayList<Box> unloadboxes0 = new ArrayList<Box>();
//		    for (int i =0;i<clients.get(0).getGoodsNum();i++) {
//		    	Box newBox = new Box();
//		    	newBox = clients.get(0).getGoods().get(i);
//		    	unloadboxes0.add(newBox);
//		    }
//		    	
//		    
//        	unloadboxes0.sort(new Comparator<Box>() {
//				/**
//				 * 按底面积从大到小+height从高到低进行排序。
//				 */
//				@Override
//				public int compare(Box b1, Box b2) {
//					// TODO Auto-generated method stub
//					double volume1=b1.getLength()*b1.getWidth();
//					double volume2=b2.getLength()*b2.getWidth();
//					if(volume1>volume2)
//						return -1;
//					else if (volume1<volume2)
//						return 1;
//					else
//						if(b1.getHeight()>b2.getHeight())
//							return -1;
//						else if(b1.getHeight()<b2.getHeight())
//							return 1;
//						else
//							return 0;
//				}
//        	});
//		    
//		    unloadboxes0.get(0).setHeight(100);
//    		System.out.print(unloadboxes0.get(0).getHeight());
//    		System.out.print(clients.get(0).getGoods().get(0).getHeight());

		    
		    
		    for (int i = 0; i < NUM_nodes; i++) {
		    	int nk =0;
		    	while (client_v_w[i][0]>1.1*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
		    		client_v_w[i][0] -= truck_weight_ratio[i]*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1];
		    		
		    		Route route = new Route((int)(1000+Math.random()*8999));
		    		
		        	ArrayList<Box> unloadboxes = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_trans = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_half_trans = new ArrayList<Box>();
		        	
		        	double volume_precent = 0.0;
		        	int ngoods = clients.get(i).getGoodsNum();
		        	//ArrayList<Box> goods = new ArrayList<Box>();
		        	for (int k = nodes_record[i]; k < ngoods; k++) {
		        		volume_precent += clients.get(i).getGoods().get(k).getVolume();
		        		if( volume_precent >= truck_weight_ratio[i]*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
		        			nodes_record[i] = k;
				        	//nk = k;
		        			break;
		        		}
		        		//goods.add(clients.get(node_id).getGoods().get(k));
		        		unloadboxes.add(clients.get(i).getGoods().get(k));
		        		unloadboxes_trans.add(clients_trans.get(i).getGoods().get(k));
		        		unloadboxes_half_trans.add(clients_half_trans.get(i).getGoods().get(k));
		        		//client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
		        	}
		        	
        	
		        	LinkedList<Node> nodes = new LinkedList<Node>();
		        	
		        	nodes.add(new Node(depot_start));
		        	nodes.add(new Node(clients.get(i)));
		        	nodes.add(new Node(depot_end));
		        	
		        	route.setNodes(nodes);
		        	
		        	ArrayList<Integer> k = null;
		        	
			        for (int j = 0; j < Type_trucks; j++) {
						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(j));
						route.setCarriage(vehicle1);	
						int if_pack = 1;
						double unload_v = 0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								if_pack = 0;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
						}
						if (unload_v > vehicle1.getTruckVolume()) {
							if_pack = 0;
						}
						if (if_pack == 1) {
							k = route.zqlbpp(unloadboxes,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_half_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_half_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
						}
						n_3DPacking += 1;
				        if(n_3DPacking>=2000) {
				        	break;
				        }
			        }
			       // System.out.println( k.size());
			       // System.out.println(unloadboxes.size());
			        
			        if (k!=null) {
				        nk += k.size();
				        solution.getRoutes().add(route);
				        solution.evaluation();
			        }
    		
		    	}
		    	n_loadboxes += nk;
		    	if  (client_v_w[i][0] >  split_minv) {
		        	new_num_nodes += 2; 
		    	}
		    	else {
		        	new_num_nodes += 1; 	
		    	} 	
		    }
		    

		    
			numberOfCities_ = new_num_nodes;
			int[] client_city_map = new int[numberOfCities_];
			     
			double[][] new_node_coords = new double[numberOfCities_][2];
			double[] new_node_v = new double[numberOfCities_];
			
			
			int var = 2;
			new_num_nodes = 0;
			for (int i = 0; i < NUM_nodes; i++) {
				if  (client_v_w[i][0] >   split_minv) {
					String s = aSolutionsList_.getDecisionVariables()[var].toString() ; 
				    char [] s0 = s.toCharArray();
				    double s00 = 0;
				    for (int j=0;j<2;j++) {
				    	s00 += (s0[j]- 48)*Math.pow(2, j);
				    }
				    client_city_map[new_num_nodes] = i;
			    	new_node_v[new_num_nodes] = client_v_w[i][0]*(s00/Split_precision);
			    	new_num_nodes += 1; 
			    	client_city_map[new_num_nodes] = i;
			    	new_node_v[new_num_nodes] = client_v_w[i][0]*(1.0-s00/Split_precision);
			    	new_num_nodes += 1; 
					
					var += 1;
				}
				else {
					client_city_map[new_num_nodes] = i;
					new_node_v[new_num_nodes] = client_v_w[i][0];    	
					new_num_nodes += 1; 	
				}
			}
		        
		    no_results += 1;
		    	    
			String if_start = aSolutionsList_.getDecisionVariables()[1].toString();
			char [] if_start0 = if_start.toCharArray();
			
			//int[] route_use_large = new int[numberOfCities_];
			double[] route_hard = new double[numberOfCities_];
			int nroute = 0;
		    for (int i = 0; i < (numberOfCities_ ); i++) {
		    	route_hard[i] = 1.0;

		        int x1 ; 
		        x1 = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[i] ;  
		        if(i>0) {
		            if (  (if_start0[i-1]- 48) == 1) {
		            	nroute += 1;
		            }
		        }
//		        if ( if_large_box[new_node_no[x]-1]==1) {
//		        	route_use_large[nroute] = 1;
//		        }
		        if (if_hard_node[client_city_map[x1]]<route_hard[nroute]) {
		        	route_hard[nroute] = if_hard_node[client_city_map[x1]];
		        }
		        //route_hard[nroute] =  route_hard[nroute]*if_hard_node[new_node_no[x]-1];
		    }
			

	        int n_route = 0;
	        double[] w_route = new double[100];
	        int[][] route_node_map = new int[100][100];
	        int[] n_node = new int[100];
	        for (int i = 0; i < 100; i++) {
	        	 n_node[i] = 0;
	        	 w_route[i] = 0.0;
	        }
	        for (int j = 0; j < numberOfCities_; j++) {
	      	  
		      	x = ((Permutation)aSolutionsList_.getDecisionVariables()[0]).vector_[j];
		      	if (j>0 && (if_start0[j-1]-48)==1) {
				          n_route += 1;
		      	}
		        w_route[n_route] += new_node_v[x];
		        route_node_map[n_route][n_node[n_route]] = x;
		        n_node[n_route] += 1;
	        }
	        
	        int[] node_if_check = new int[NUM_nodes];
	        for (int i = 0; i < NUM_nodes; i++) {
	        	node_if_check[i] = 0;
	        }
				        
	        for (int i = 0; i < n_route+1; i++) {
	        	
		        ArrayList <Integer> sort_check = new ArrayList<Integer>();
		        int[][] sort_nodes0 = new int[n_node[i]][2];
		        for (int ss= 0;ss<n_node[i];ss++) {
		        	sort_nodes0[ss][0]=100;
		        	sort_nodes0[ss][1]=100;
		        }
		        //sort_nodes0 = route_node_map[i];
		        int nsort = 0;
		        for (int j = 0; j < n_node[i]; j++) {
		        	
		        	int new_node_id = route_node_map[i][j];
		        	int node_id = client_city_map[new_node_id];
		        	
	        		int id_add0 = 0;
	        		int ifadd = 1;
	        		if (sort_check != null) {
	        			id_add0 = sort_check.size();
	        			
	        			for (int ss =0;ss<sort_check.size();ss++) {
	        				if(sort_check.get(ss)==node_id) {
	        					ifadd = 0;
	        					id_add0 = ss;
	        				}	
	        			}
	        		}
	        		if(ifadd==1) {
	        			sort_check.add(node_id);
	        		}
		        	if(sort_nodes0[id_add0][0]==100) {
		        		sort_nodes0[id_add0][0] = new_node_id;
		        	}
		        	else {
		        		sort_nodes0[id_add0][1] = new_node_id;
		        	}
		        }
		        int nj = 0;
		        for (int ss = 0; ss < n_node[i]; ss++ ){
		        	if (sort_nodes0[ss][0]!=100) { 
		        		route_node_map[i][nj] = sort_nodes0[ss][0];
		        		nj+=1;
		        	}
		        	if (sort_nodes0[ss][1]!=100) { 
		        		route_node_map[i][nj] = sort_nodes0[ss][1];
		        		nj+=1;
		        	}
		        }
		        //System.out.println(n_node[i]);
		        //System.out.println(nj);
	        	
		        //ArrayList<Node> node_mid =  new ArrayList<Node>();
	        	int current_truck_id = 0;
		        Route route = new Route((int)(1000+Math.random()*8999));
		        for (int j = 0; j < Type_trucks; j++) {
		        	if (w_route[i] < route_hard[i]*VEHICLE_VOLUME[j]*relax_ratio) {
		        		current_truck_id = j;
		        		break;
		        	}
		        	current_truck_id = j;
		        }
				Carriage vehicle = new Carriage(BASIC_TRUCKS.get(current_truck_id));
				route.setCarriage(vehicle);	
				LinkedList<Node> nodes = new LinkedList<Node>();
				ArrayList<Box> unloadboxes = new ArrayList<Box>();
				ArrayList<Box> unloadboxes_trans = new ArrayList<Box>();
				ArrayList<Box> unloadboxes_half_trans = new ArrayList<Box>();
				
				nodes.add(new Node(depot_start));
				
				int[] node_id_last = new int[n_node[i]];
				for(int nn=0;nn<n_node[i];nn++) {
					node_id_last[nn] =100;
				}
//				if(nj==4) {
//					System.out.println("start");
//				}
				
		        for (int j = 0; j < n_node[i]; j++) {
		        	
		        	
		        	ArrayList<Box> unloadboxes_node = new ArrayList<Box>();
		        	ArrayList<Box> unloadboxes_node_trans = new ArrayList<Box>();
		        	ArrayList<Box> unloadboxes_node_half_trans = new ArrayList<Box>();
		        	int new_node_id = route_node_map[i][j];
		        	int node_id = client_city_map[new_node_id];
		        	
		        	
		        	if(new_node_v[new_node_id]>0.1) {
		        	
			        	int ngoods = clients.get(node_id).getGoodsNum();
			        	//System.out.println(node_id);
			        	//node_mid.add(new Node(clients.get(node_id)));
			        	if (j>0) {
			        		int if_add = 1;
				        	for (int node_check = 0; node_check < j; node_check++) {
				        		if (node_id == node_id_last[node_check]) {
				        			if_add = 0;
	//			        			if(node_check!=j-1) {
	//			        				System.out.println("wrong");
	//			        			}
				        		}
				        	}
			        		if(if_add == 1) {
				        		//if(new_node_v[new_node_id]>0.1) {
				        			nodes.add(new Node(clients.get(node_id)));  
				        			//node_id_last[j] = node_id;
				        		//}
			        		}
			        	}
			        	else {
			        		//if(new_node_v[new_node_id]>0.1) {
			        			nodes.add(new Node(clients.get(node_id))); 
			        			//node_id_last[j] = node_id;
			        		//}
			        		      		
			        	}
	
			        	node_id_last[j] = node_id;
			        	//int ngoods = clients.get(node_id).getGoodsNum();
			        	double volume_precent = 0.0;
			        	//ArrayList<Box> goods = new ArrayList<Box>();
			        	for (int k = node_if_check[node_id]+nodes_record[node_id]; k < ngoods; k++) {
			        		volume_precent += clients.get(node_id).getGoods().get(k).getVolume();
			        		if(k>node_if_check[node_id]+nodes_record[node_id]) {
				        		if(node_if_check[node_id] == 0 && volume_precent > new_node_v[new_node_id]) {
				        			node_if_check[node_id] = k-nodes_record[node_id];
		//		        			if (node_if_check[node_id]==0) {
		//		        				nodes.remove(new Node(clients.get(node_id)));
		//		        			}
				        			break;
				        		}
			        		}
			        		//goods.add(clients.get(node_id).getGoods().get(k));
			        		unloadboxes_node.add(clients.get(node_id).getGoods().get(k));
	//		        		Box a = clients.get(node_id).getGoods().get(k);
	//		        		Box b = clients_trans.get(node_id).getGoods().get(k);
			        		unloadboxes_node_trans.add(clients_trans.get(node_id).getGoods().get(k));
			        		unloadboxes_node_half_trans.add(clients_half_trans.get(node_id).getGoods().get(k));
			        	}
			        	//node_mid.get(j).setGoods(goods);
	//		        	unloadboxes_node.sort(new Comparator<Box>() {
	//						/**
	//						 * 按底面积从大到小+height从高到低进行排序。
	//						 */
	//						@Override
	//						public int compare(Box b1, Box b2) {
	//							// TODO Auto-generated method stub
	//							double volume1=b1.getLength()*b1.getWidth();
	//							double volume2=b2.getLength()*b2.getWidth();
	//							if(volume1>volume2)
	//								return -1;
	//							else if (volume1<volume2)
	//								return 1;
	//							else
	//								if(b1.getHeight()>b2.getHeight())
	//									return -1;
	//								else if(b1.getHeight()<b2.getHeight())
	//									return 1;
	//								else
	//									return 0;
	//						}
	//		        	});
			        	for (int n_unload=0;n_unload<unloadboxes_node.size();n_unload++) {
			        		unloadboxes.add(unloadboxes_node.get(n_unload));
			        		unloadboxes_trans.add(unloadboxes_node_trans.get(n_unload));
			        		unloadboxes_half_trans.add(unloadboxes_node_half_trans.get(n_unload));
			        	}
		        	}
		        }

				nodes.add(new Node(depot_end));				
				route.setNodes(nodes);

//				ArrayList<Integer> k1;//k里面的boxes下标都已经装载了。
//				ArrayList<Integer> k2;
//				ArrayList<Integer> k3;
//				ArrayList<Integer> k4;
//				ArrayList<Integer> k5;
				double volume_route = 0;
				for (int unloadbox = 0; unloadbox < unloadboxes.size(); unloadbox++) {
					volume_route += unloadboxes.get(unloadbox).getVolume();
				}
				
				ArrayList<Integer> k = null;
				
				// 小车换大车
				
				if (unloadboxes.size() >0) {
			        for (int j = 0; j < Type_trucks; j++) {
						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(j));
						route.setCarriage(vehicle1);	
						int if_pack = 1;
						double unload_v = 0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								if_pack = 0;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
						}
						if (unload_v > vehicle1.getTruckVolume()) {
							if_pack = 0;
						}
						if (if_pack == 1) {
							k = route.zqlbpp(unloadboxes,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_half_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_half_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
						}
						
			        }
			                

					if (k!=null) {
						n_loadboxes += k.size();
						solution.getRoutes().add(route);
						solution.evaluation();
						//System.out.println(k.size());
						//System.out.println(unloadboxes.size());
						if (k.size() !=unloadboxes.size()) {
							break;
						}
					}
					n_3DPacking += 1;
			        if(n_3DPacking>=2000) {
			        	break;
			        }
	
				}


//				if (k != null) {
//					if (k.size() != unloadboxes.size()) {
//						break;
//					}
//				}

				
//				if(k.size() == unloadboxes.size()) {
//					//System.out.println("F1 = "+solution.getF1());
//					//System.out.println("F2 = "+solution.getF2());	
//				}

	        }
	        
	        
	        int n_all_boxs = 0;
	        for (int n_client = 0;n_client<NUM_nodes;n_client++) {
	        	n_all_boxs += clients.get(n_client).getGoodsNum();
	        }
	        

	        if (n_loadboxes == n_all_boxs) {
	        	solutionSet.add(new Solution_vrp(solution));
	        	save_feasible_no[n_sol] = 1;        	
	        }
	        

	        //n_solutions += 1;
	        
	    }
	


	return solutionSet;
	}
	
	
	static SolutionSet_vrp get_Solutions_naive(int file_id, double[] truck_weight_ratio0) throws IOException {

		HashMap <String, Double> Distances = new HashMap<String, Double>();		
		HashMap Data = new HashMap();
		ArrayList<Carriage> BASIC_TRUCKS = new ArrayList<Carriage>();
		Data = input(file_id);
		Distances  = (HashMap<String, Double>) Data.get("distances");
		BASIC_TRUCKS  = (ArrayList<Carriage>) Data.get("trucks");
		int Trucktype_num = (int) Data.get("trucktype_num");
		int NUM_nodes = (int) Data.get("client_num");
		double[] VEHICLE_CAPACITY = new double[Trucktype_num];
		double[] VEHICLE_VOLUME = new double[Trucktype_num];
		
		for (int i=0; i<Trucktype_num; i++ ) {
			VEHICLE_CAPACITY[i] = (double) BASIC_TRUCKS.get(i).getCapacity();
			VEHICLE_VOLUME[i] = (double) BASIC_TRUCKS.get(i).getTruckVolume();
		}

		double[][] client_v_w = (double[][]) Data.get("client_v_w");
		ArrayList<Node> clients =  (ArrayList<Node>) Data.get("clients");
		ArrayList<Node> clients_trans =  (ArrayList<Node>) Data.get("clients_trans");
		ArrayList<Node> clients_half_trans =  (ArrayList<Node>) Data.get("clients_half_trans");
		Node depot_start = (Node) Data.get("depot_start");
		Node depot_end = (Node) Data.get("depot_end");
		
		//double relax_volume = relax_ratio*VEHICLE_VOLUME[0];
	  
	  
	    int Type_trucks = VEHICLE_CAPACITY.length;
	    double Split_precision = 6.0;
	    int no_results = 0;
	    int x = 0 ;
	    //int numberOfVariables = solutionsList_0.get(0).getDecisionVariables().length ;
	    //int n_solutions = 0;
	    double[] truck_weight_ratio = truck_weight_ratio0;
	    double truck_w_r_min = 0.6;
	    for (int i=0;i<NUM_nodes;i++) {
	    	if(truck_w_r_min<truck_weight_ratio[i]) {
	    		truck_weight_ratio[i]=truck_w_r_min;
	    	}
	    }

	    SolutionSet_vrp solutionSet = new SolutionSet_vrp();

	    	
	    	Solution aSolutionsList_ ;
	
		    Solution_vrp solution = new Solution_vrp();
		    solution.distanceMap=Distances;
	      	
	        int n_loadboxes = 0;
	        int n_loadboxes_check = 0;
		    int new_num_nodes = 0;
		    int numberOfCities_ = 0;
		    
		    int[] nodes_record = new int[NUM_nodes];
	  
		    for (int i = 0; i < NUM_nodes; i++) {
		    	if (truck_weight_ratio[i]==1) {
		    		truck_weight_ratio[i] = truck_w_r_min;
		    	}
		    	int nk =0;
		    	while (client_v_w[i][0]>0.01*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
		    		
		    		
		    		Route route = new Route((int)(1000+Math.random()*8999));
		    		
		        	ArrayList<Box> unloadboxes = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_trans = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_half_trans = new ArrayList<Box>();
		        	
		        	double volume_precent = 0.0;
		        	int ngoods = clients.get(i).getGoodsNum();
		        	//ArrayList<Box> goods = new ArrayList<Box>();
		        	for (int k = nodes_record[i]; k < ngoods; k++) {
		        		volume_precent += clients.get(i).getGoods().get(k).getVolume();
		        		if( volume_precent >= truck_weight_ratio[i]*VEHICLE_VOLUME[VEHICLE_CAPACITY.length-1]) {
		        			nodes_record[i] = k;
				        	//nk = k;
		        			break;
		        		}
		        		//goods.add(clients.get(node_id).getGoods().get(k));
		        		unloadboxes.add(clients.get(i).getGoods().get(k));
		        		unloadboxes_trans.add(clients_trans.get(i).getGoods().get(k));
		        		unloadboxes_half_trans.add(clients_half_trans.get(i).getGoods().get(k));
		        		client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
		        		//client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
		        	}
		        	
	    	
		        	LinkedList<Node> nodes = new LinkedList<Node>();
		        	
		        	nodes.add(new Node(depot_start));
		        	nodes.add(new Node(clients.get(i)));
		        	nodes.add(new Node(depot_end));
		        	
		        	route.setNodes(nodes);
		        	
		        	ArrayList<Integer> k = null;
		        	
			        for (int j = Type_trucks-1; j < Type_trucks; j++) {
						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(j));
						route.setCarriage(vehicle1);	
						int if_pack = 1;
						double unload_v = 0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								if_pack = 0;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								if_pack = 0;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
						}
						if (unload_v > vehicle1.getTruckVolume()) {
							if_pack = 0;
						}
						if (if_pack == 1) {
							k = route.zqlbpp(unloadboxes,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.zqlbpp(unloadboxes_half_trans,2);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
							k = route.dblf(unloadboxes_half_trans,1);
							//n_3DPacking += 1;
							if (k.size() == unloadboxes.size()) {
								break;
							}
							//System.out.println(k.size());
						}
						n_3DPacking += 1;
				        if(n_3DPacking>=2000) {
				        	break;
				        }
			        }
			       // System.out.println( k.size());
			       // System.out.println(unloadboxes.size());
			        
			        if (k!=null) {
				        nk += k.size();
				        solution.getRoutes().add(route);
				        solution.evaluation();
			        }
			
		    	}
		    	n_loadboxes+=nk;
 	
		    }
		    

	        int n_all_boxs = 0;
	        for (int n_client = 0;n_client<NUM_nodes;n_client++) {
	        	n_all_boxs += clients.get(n_client).getGoodsNum();
	        }
			//System.out.println(n_loadboxes);
			//System.out.println(n_all_boxs);

	        if (n_loadboxes == n_all_boxs) {
	        	solutionSet.add(new Solution_vrp(solution)); 	
	        }
       	

	return solutionSet;
	}
	
} // MOEAD_main

