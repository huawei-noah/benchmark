package com.my.vrp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.my.vrp.param.L;
import com.my.vrp.utils.PseudoRandom;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.JSONParser;

import feasibilityCheck.Check;


import jmetal.core.*;
import jmetal.encodings.variable.Permutation;
import jmetal.metaheuristics.moead.cMOEAD;
import jmetal.operators.crossover.CrossoverFactory;
import jmetal.operators.mutation.MutationFactory;
import jmetal.problems.CVRP_mix_integer;
import jmetal.util.JMException;
import jmetal.util.fast_nondom;
/**
 * 程序入口 算法步骤： 首先用节约算法生成初始解 生成初始解的过程需要调用装箱子程序判断能否装下
 * 
 * @author Qingling ZHU
 *
 */

public class Track1 {
	static final boolean debug = false;// 是否调试。
	static final boolean test = false;//
	static final boolean isUpdateExtreme = false;
	static final boolean isLog = false;
	static final boolean isOutput = false;
	static final boolean is_calculate_hv = false;
	
	
	//********************************PARAMETERS FOR TRACK 1
	static final int track = 1;
	static final double PFRS_TIME_LIMITS = 150;// packing first routing second time limit
	static final double RFPS_TIME_LIMITS = 500;// routing first packing second time limit
	static final double SA_TIME_LIMITS = 600;// sa time limit (not use in track 1)
	static final int NUM_COMBINATION = 50;// not used in track 1.
	static final int MAX_VRP = 10;
	static final int NUM_ROUTE_GREEDY_TEST = 10;
	static final int MOVE_BOX_BRUTEFORCE = 11;
	static final int MOVE_NODE_BRUTEFORCE = 11;
	static final int GREEDY_ITERATION = 10;
	static final int NUM_COMBINATION_MOVE_BOXES = 5;
	static final int NUM_COMBINATION_MOVE_NODES = 5;
	static final double BIT_FLIP_MOVE_BOXES = 0.1;
	static final double MAX_RATIO = 1.0;
	static final boolean CHECKSOLUTION = false;
	static final double FINAL_TIME = 600;
	static final boolean PACKING_RECORD = false;
	static final int PACKING_LIMITS = 2000;
	static final int POPSIZE = 10;
	static final int T_ = 5;
	static final int maxE = 500;
	static final int n_repeat = 1000;
	//********************************PARAMETERS FOR TRACK 1
	static int n_3DPacking;
	static int TRUCKTYPE_NUM;
	static int CLIENT_NUM;
	static int BOX_NUM;
	static ArrayList<Carriage> BASIC_TRUCKS;
	static HashMap<String, Double> distanceMap;
	static Node depot_start ;
	static Node depot_end ;
	static double[] VEHICLE_CAPACITY ;
	static double[] VEHICLE_VOLUME ;
	static double[][] client_volume_weight;
	static ArrayList<Node> clients ;
	static ArrayList<Node> clients_trans ;
	static ArrayList<Node> clients_half_trans ;
	static ArrayList<Node> clients_v ; //sort boxes by volume
	static ArrayList<Node> clients_trans_v ; //sort boxes by volume
	static ArrayList<Node> clients_half_trans_v; //sort boxes by volume
	static HashMap<Integer, String> PlatformIDCodeMap;
	static long begintime;
	// ===================================================== main procedure
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException, ClassNotFoundException, JMException {
		String input_directory = args[0];
		String output_directory = args[1];
		double max_single_time = 0.0;
		
		Problem problem;
		Algorithm algorithm;
		Operator crossover ;
		Operator mutation;
		Operator mutation_mod;
		HashMap  parameters ; // Operator parameters
	    HashMap  results;
//	    HashMap <String, Double> Distances = new HashMap<String, Double>();
	    
		/**
		 * 初始化ideal和nadir points.用来计算hv
		 */
		// 如果极点文件（存两个函数极限小值的文件）存在，则导入数据。
		Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();// problemName->ideal&nadir
		if (is_calculate_hv) {
			File f = new File("./data/extremes");
			if (f.exists()) {
				idealNadirMap = readExtreme();
			} else {
				// 如果不存在，则都设置初始值。
				f = new File(input_directory);
				String[] filenames = f.list();
				for (int fileidx = 0; fileidx < filenames.length; fileidx++) {
					Double[] initialPoints = { Double.MAX_VALUE, Double.MAX_VALUE, Double.MIN_VALUE, Double.MIN_VALUE };
					idealNadirMap.put(filenames[fileidx], initialPoints);
				}
			}
		}

		File f = new File(input_directory);
		String[] filenames = f.list();
		// 统计信息。
		double total_hv = 0.0;
		double total_time = 0.0;
  
		// 保存所有搜索到的最好解。
		SolutionSet_vrp final_ss = new SolutionSet_vrp();
		System.out.println(new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date()));
		/**
		 * 选择一个问题。
		 */
		for (int fileidx = 0; fileidx < filenames.length; fileidx++) {
			/**
			 * 用来计算hv的变量
			 */
			Double[] idealNadir;
			Random rand = new Random(System.currentTimeMillis());
			/**
			 * 初始化变量的值。
			 */
			
			begintime = System.nanoTime();// 开始计时
			if (is_calculate_hv)
				idealNadir = (Double[]) idealNadirMap.get(filenames[fileidx]);
			CLIENT_NUM = -1;//
			BOX_NUM = 0;//
			TRUCKTYPE_NUM = -1;// 汽车的数量
			PlatformIDCodeMap = new HashMap<Integer, String>();// platform ID map to code
			HashMap<String, Integer> PlatformCodeIDMap = new HashMap<String, Integer>();// platform Code map to ID

			BASIC_TRUCKS = new ArrayList<Carriage>();
			distanceMap = new HashMap<String, Double>();

			depot_start = new Node();// 起始节点
			clients = new ArrayList<Node>();
			clients_trans = new ArrayList<Node>();
			clients_half_trans = new ArrayList<Node>();
			clients_v = new ArrayList<Node>();
			clients_trans_v = new ArrayList<Node>();
			clients_half_trans_v = new ArrayList<Node>();
			
			depot_end = new Node();// 结束节点。
//			double[][] client_volume_weight=null;
			/**
			 * 开始读取输入文件的问题信息。
			 */
			// JSON parser object to parse read file
			JSONParser jsonParser = new JSONParser();
			// String instanceName = "E1594518281316";
			try (FileReader reader = new FileReader(input_directory + '/' + filenames[fileidx])) {
				JSONObject obj = (JSONObject) jsonParser.parse(reader);// 最顶层
				Iterator<JSONObject> iterator;// 用来遍历JSONObject的Array
				JSONObject algorithmBaseParamDto = (JSONObject) obj.get("algorithmBaseParamDto");
				JSONArray platformDtoList = (JSONArray) algorithmBaseParamDto.get("platformDtoList");
				CLIENT_NUM = platformDtoList.size();
				boolean[] mustFirst = new boolean[CLIENT_NUM];
				client_volume_weight = new double[CLIENT_NUM][2];
				for(int i=0;i<CLIENT_NUM;i++) {
					client_volume_weight[i][0] = 0.0;
					client_volume_weight[i][1] = 0.0;
				}
				// PlatformMap，文件中platform和代码中platformID的对应关系。
				iterator = platformDtoList.iterator();
				int platformID = 0;
				PlatformIDCodeMap.put(platformID, "start_point");
				PlatformCodeIDMap.put("start_point", platformID);
				platformID++;
				while (iterator.hasNext()) {
					JSONObject platform = iterator.next();
					String platformCode = (String) platform.get("platformCode");
					PlatformIDCodeMap.put(platformID, platformCode);
					PlatformCodeIDMap.put(platformCode, platformID);
					mustFirst[platformID - 1] = (boolean) platform.get("mustFirst");
					platformID++;
				}
				PlatformIDCodeMap.put(platformID, "end_point");
				PlatformCodeIDMap.put("end_point", platformID);
				// 得到各种类别的卡车
				JSONArray truckTypeDtoList = (JSONArray) algorithmBaseParamDto.get("truckTypeDtoList");
				TRUCKTYPE_NUM = truckTypeDtoList.size();
				for (int basic_truct = 0; basic_truct < TRUCKTYPE_NUM; basic_truct++) {
					Carriage truck = new Carriage();
					JSONObject curr_truck = (JSONObject) truckTypeDtoList.get(basic_truct);

					truck.setCapacity((double) curr_truck.get("maxLoad"));
					truck.setHeight((double) curr_truck.get("height"));
					truck.setLength((double) curr_truck.get("length"));
					truck.setWidth((double) curr_truck.get("width"));
					truck.setTruckId(basic_truct);
					truck.setTruckTypeId((String) curr_truck.get("truckTypeId"));
					truck.setTruckTypeCode((String) curr_truck.get("truckTypeCode"));
					BASIC_TRUCKS.add(truck);
				}

				// 读取distanceMap
				JSONObject distanceMapJSON = (JSONObject) algorithmBaseParamDto.get("distanceMap");
				for (int clienti = 1; clienti <= CLIENT_NUM; clienti++) {
					for (int clientj = 1; clientj <= CLIENT_NUM; clientj++) {
						if (clienti != clientj) {
							// 不同的client之间的距离从文件里面读取。
							String twoplatforms = PlatformIDCodeMap.get(clienti) + '+' + PlatformIDCodeMap.get(clientj);
							// String twoplatform = clients.get(clienti).getPlatformID());
							distanceMap.put(String.valueOf(clienti) + '+' + String.valueOf(clientj),
									(Double) distanceMapJSON.get(twoplatforms));
						} else {
							// 相同的client之间的距离为0.
							distanceMap.put(String.valueOf(clienti) + '+' + String.valueOf(clientj), 0.0);
						}
					}
				}
				// 从起始点到每个client的距离。
				for (int clienti = 1; clienti <= CLIENT_NUM; clienti++) {
					String twoplatforms = PlatformIDCodeMap.get(0) + '+' + PlatformIDCodeMap.get(clienti);
					distanceMap.put(String.valueOf(0) + '+' + String.valueOf(clienti),
							(Double) distanceMapJSON.get(twoplatforms));
				}
				// 从每个client到终点的距离。
				for (int clienti = 1; clienti <= CLIENT_NUM; clienti++) {
					String twoplatforms = PlatformIDCodeMap.get(clienti) + '+' + PlatformIDCodeMap.get(CLIENT_NUM + 1);
					distanceMap.put(String.valueOf(clienti) + '+' + String.valueOf(CLIENT_NUM + 1),
							(Double) distanceMapJSON.get(twoplatforms));
				}

				// 最后读取boxes，初始化start_point, clients, end_point
				depot_start.setPlatformID(0);// start no. is always 0
				depot_start.setDemands(0);// start demands is always 0
				depot_start.setGoodsNum(0);// start goodsNum is always 0
				depot_start.setGoods(new ArrayList<Box>());//
				depot_start.setMustFirst(false);

				depot_end.setPlatformID(CLIENT_NUM + 1);//
				depot_end.setDemands(0);
				depot_end.setGoodsNum(0);
				depot_end.setGoods(new ArrayList<Box>());
				depot_end.setMustFirst(false);

				// 建立clients所有客户，没有箱子需求。
				for (int i = 1; i <= CLIENT_NUM; i++) {
					Node client = new Node();
					ArrayList<Box> boxes = new ArrayList<Box>();
					int platform = i;
					client.setPlatformID(platform);// 第几个平台，用于distance-matrix的下标。
					client.setDemands(0);// demands==0,the client's demands are boxes
					client.setGoods(boxes);
					client.setGoodsNum(0);// goods num
					client.setLoadgoodsNum(0);// 刚开始所有boxes都没有装载。
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
				// 遍历一遍boxes，为客户添加box
				JSONArray boxesJSONArray = (JSONArray) obj.get("boxes");
				iterator = boxesJSONArray.iterator();
				int if_half = 0;
				while (iterator.hasNext()) {
					JSONObject currBoxJSON = iterator.next();
					String platformCode = (String) currBoxJSON.get("platformCode");
					// 找到它属于哪个客户ID
					platformID = PlatformCodeIDMap.get(platformCode);
					Box box = new Box();
					box.setSpuBoxID((String) currBoxJSON.get("spuBoxId"));// spuBoxId,specific unique box id
					box.setPlatformid(platformID);
					box.setHeight((double) currBoxJSON.get("height"));// height
					double width = (double) currBoxJSON.get("width");
					double length = (double) currBoxJSON.get("length");
					box.setWidth(width);// width
					box.setLength(length);// length
					box.setWeight((double) currBoxJSON.get("weight"));// fragile or not
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
					// if(width>length) {
					// box.setDirection(200);
					//// System.out.println();
					//// System.out.println(filenames[fileidx]);
					// }else {
					box.setDirection(100);
					box_trans.setDirection(100);
					box_half_trans.setDirection(100);
					// }
					BOX_NUM += 1;
					// 为这个客户添加当前的box
					clients.get(platformID - 1).getGoods().add(box);
					clients_trans.get(platformID-1).getGoods().add(box_trans);
					clients_half_trans.get(platformID-1).getGoods().add(box_half_trans);
					//其他信息
					client_volume_weight[platformID-1][0] = client_volume_weight[platformID-1][0] + box.getWidth()*box.getLength()*box.getHeight();
					client_volume_weight[platformID-1][1] = client_volume_weight[platformID-1][1] + box.getWeight();
				}
				// 最后设置GoodsNum和mustfirst
				for (int i = 0; i < CLIENT_NUM; i++) {
					clients.get(i).setGoodsNum(clients.get(i).getGoods().size());// goods num
					clients.get(i).setMustFirst(mustFirst[i]);
					clients_trans.get(i).setGoodsNum(clients_trans.get(i).getGoods().size());//goods num
					clients_trans.get(i).setMustFirst(mustFirst[i]);
					clients_half_trans.get(i).setGoodsNum(clients_half_trans.get(i).getGoods().size());//goods num
					clients_half_trans.get(i).setMustFirst(mustFirst[i]);
				}

				/**
				 * 计算每个节点需要的载重和体积。
				 */
				for (int i = 0; i < CLIENT_NUM; i++) {
					double weight_sum = 0.0;
					double volumn_sum = 0.0;
					for (Box b : clients.get(i).getGoods()) {
						weight_sum = weight_sum + b.getWeight();
						volumn_sum = volumn_sum + b.getVolume();
					}
					clients.get(i).setGoodsWeight(weight_sum);
					clients.get(i).setGoodsVolumn(volumn_sum);
				}

				BASIC_TRUCKS.sort(new Comparator<Carriage>() {
					public int compare(Carriage c1, Carriage c2) {
						double d1 = c1.getTruckVolume();
						double d2 = c2.getTruckVolume();
						if (d1 < d2)
							return -1;
						else if (d1 > d2)
							return 1;
						else
							return 0;
					}
				});
				
				//to get clients_v, clients_trans_v, clients_half_trans_v
				for(int i=0;i<CLIENT_NUM;i++) {
					clients_v.add(new Node(clients.get(i)));
					clients_trans_v.add(new Node(clients_trans.get(i)));
					clients_half_trans_v.add(new Node(clients_half_trans.get(i)));
				}
				//begin sort boxes of all kinds of clients
				for(int i=0;i<CLIENT_NUM;i++) {
					clients.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
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
					clients_trans.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
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
					clients_half_trans.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
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
					clients_v.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
							double volume1=b1.getVolume();
							double volume2=b2.getVolume();
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
					clients_trans_v.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
							double volume1=b1.getVolume();
							double volume2=b2.getVolume();
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
					clients_half_trans_v.get(i).getGoods().sort(new Comparator<Box>() {
						public int compare(Box b1, Box b2) {
							double volume1=b1.getVolume();
							double volume2=b2.getVolume();
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

			// int N_TIME_STEP1 = Integer.MAX_VALUE;
			// int N_TIME_STEP2 = Integer.MAX_VALUE;
			// int N_TIME_STEP3 = Integer.MAX_VALUE;
			// int N_TIME_STEP4 = Integer.MAX_VALUE;

			int N_TIME_STEP1 = 0;
			int N_TIME_STEP2 = 0;
			int N_TIME_STEP3 = 0;
			int N_TIME_STEP4 = 0;

			N_TIME_STEP1 = Integer.MAX_VALUE;
			N_TIME_STEP2 = Integer.MAX_VALUE;

			n_3DPacking = 0;

			VEHICLE_CAPACITY = new double[TRUCKTYPE_NUM];//每輛車的載重。
			VEHICLE_VOLUME = new double[TRUCKTYPE_NUM];//每輛車的體積。
			
		    for (int i=0; i<TRUCKTYPE_NUM; i++ ) {
		    	VEHICLE_CAPACITY[i] = (double) BASIC_TRUCKS.get(i).getCapacity();
		    	VEHICLE_VOLUME[i] = (double) BASIC_TRUCKS.get(i).getTruckVolume();
		    }
		    double[][] client_v_w = new double[CLIENT_NUM][2];//client_volume_weight;
		    for(int i=0;i<CLIENT_NUM;i++) {
		    	client_v_w[i][0] = client_volume_weight[i][0];
		    	client_v_w[i][1] = client_volume_weight[i][1];
		    }
		    int[] if_large_box = new int[clients.size()];
		    double[] if_hard_node = new double[clients.size()];
		    double[] if_hard_node0 = new double[clients.size()];
		    ArrayList<Double> v_different = new ArrayList<Double>();
		    
		    for (int i=0; i<clients.size(); i++) {
				//對於節點i
				if_hard_node[i] = 1.0;
				if_hard_node0[i] = 1.0;
				double max_v_box = 0;
				int max_v_box_n = 0;
			    double[] vs = new double[10000];
			    int[] vs_no = new int[10000];

				for (int j=0; j<clients.get(i).getGoodsNum(); j++) {
					if (clients.get(i).getGoods().get(j).getHeight() > BASIC_TRUCKS.get(0).getHeight()){
						if_large_box[i] = 1;
						break;
					}
				}
				//System.out.println(clients.get(i).getDemands());
				double node_v_all = 0;//所有箱子的體積
				for (int nbox=0; nbox<clients.get(i).getGoodsNum();nbox++) {
					node_v_all += clients.get(i).getGoods().get(nbox).getVolume();
				}
				if(node_v_all>0.15*VEHICLE_VOLUME[TRUCKTYPE_NUM-1]) {
					
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
					if (max_box.getHeight()>0.5*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()&& max_box.getHeight()<0.75*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()) {
						if_hard_node[i] = 0.8;//體積最大的箱子
					}
					if (max_box.getWidth()>0.5*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()&& max_box.getWidth()<0.75*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()) {
						if_hard_node[i] = 0.8;
					}
					if (max_box.getHeight()>0.33334*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()&& max_box.getHeight()<0.4*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()) {
						if_hard_node[i] = 0.85;
					}
					if (max_box.getWidth()>0.333334*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()&& max_box.getWidth()<0.4*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()) {
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
						if (max_boxs.getHeight()>0.5*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()&& max_boxs.getHeight()<0.75*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()) {
							if_hard_node[i] = 0.8;
						}
						if (max_boxs.getWidth()>0.5*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()&& max_boxs.getWidth()<0.75*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()) {
							if_hard_node[i] = 0.8;
						}
						if (max_boxs.getHeight()>0.33334*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()&& max_boxs.getHeight()<0.4*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getHeight()) {
							if_hard_node[i] = 0.85;
						}
						if (max_boxs.getWidth()>0.333334*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()&& max_boxs.getWidth()<0.4*BASIC_TRUCKS.get(TRUCKTYPE_NUM-1).getWidth()) {
							if_hard_node[i] = 0.85;
						}
					}
				}
			}
			
			
		    /**
		     * 找到每個節點的裝載率。
		     * truck_weight_ratio[i]==1.0表示這個節點用一輛車裝夠了。
		     */
		    double[] truck_weight_ratio = new double[CLIENT_NUM];
		    for (int i = 0; i < CLIENT_NUM; i++) {
		    	truck_weight_ratio[i] = 1.0;
		    	if (client_v_w[i][0]>1.1*VEHICLE_VOLUME[TRUCKTYPE_NUM-1]) {//這個節點箱子的體積比較大。很大概率一輛車裝不完。
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
			        		if( volume_precent >= truck_weight_ratio[i]*VEHICLE_VOLUME[TRUCKTYPE_NUM-1]) {
			        			break;
			        		}
			        		//goods.add(clients.get(node_id).getGoods().get(k));
			        		unloadboxes.add(new Box(clients.get(i).getGoods().get(k)));
			        		unloadboxes_trans.add(new Box(clients_trans.get(i).getGoods().get(k)));
			        		unloadboxes_half_trans.add(new Box(clients_half_trans.get(i).getGoods().get(k)));
			        		//client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
			        	}
			        	
			        	ArrayList<Integer> k = null;

						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(TRUCKTYPE_NUM-1));//用最後一種車型。
						route.setCarriage(vehicle1);	
	
						boolean pack_checking = true;
						double unload_v = 0.0;//有多少體積沒有裝載的。
						double unload_w = 0.0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {//這個箱子不能用這個車子裝。
								pack_checking = false;break;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								pack_checking = false;break;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								pack_checking = false;break;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
							unload_w += unloadboxes.get(nbox).getWeight();
						}
						if (unload_v > MAX_RATIO*vehicle1.getTruckVolume()||
								unload_w > MAX_RATIO*vehicle1.getCapacity()) {//如果要裝載的箱子的體積更大，也不用pack了。
							pack_checking = false;
						}
						if (pack_checking) {
							n_3DPacking += 1;
							k = route.is_loadable(unloadboxes);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_half_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
						}
		    		}
		    	}//if 
		    }//for each node
		    		

		    
//		    // main settings *************************************************
//		    int n_repeat = 15;
//		    int n_split = 4;
//		    double[] relax_ratio_all = {0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55,0.525,0.5,0.475,0.45,0.425,0.4,0.375,0.35};
//		    // main settings finished *************************************************
		    
		    //double[][] save_coords = new double[n_repeat*n_split][2] ;
		    HashMap PF_out = new HashMap();
		    
		    
		    //ArrayList <double[]> save_all_tested_objs = new ArrayList <double[]>();
	    	ArrayList<double[]> best_results = new ArrayList <double[]>();
		    
//		    SolutionSet population_last = null;
//		    SolutionSet_vrp solutionSet_last = new SolutionSet_vrp();
		    
		    ///////////end
			/**
			 * 读取信息结束。 程序开始。
			 */

			System.out
					.print(""+fileidx+"\t"+filenames[fileidx] + "\t" + CLIENT_NUM + "\t" + BOX_NUM + "\t" + TRUCKTYPE_NUM + "\t");
			ArrayList<Node> curr_clients = new ArrayList<Node>();
			if(true) {
				for (int box_method = 1; box_method <= 6; box_method++) {
					if(box_method==1) {
						curr_clients = clients;
					} else if (box_method==2) {
						curr_clients = clients_trans;
					} else if (box_method==3) {
						curr_clients = clients_half_trans;
					} else if (box_method==4) {
						curr_clients = clients_v;
					} else if (box_method==5) {
						curr_clients = clients_trans_v;
					} else if (box_method==6) {
						curr_clients = clients_half_trans_v;
					} 
					for (int pack_method = 0; pack_method <= 4; pack_method++) {
						SolutionSet_vrp solutionSet = new SolutionSet_vrp();
						/**
						 * Track 1 procedure heuristic method to vrp method. Q： 对于每一个平台的箱子，用什么车子来装？A：
						 * 装载率最高的车子装。 Q： 如果这个平台的箱子，一个车子装不下怎么办？A： 截断装载，注意后面一步分判断是否可以和前面的车子进行结合。 Q：
						 * 如果这个平台的箱子可以一个车子装下，则判断是否可以和前面的车子进行结合，如果可以，则结合。如果不可以，则创建新的车子。
						 */
						// clients: store the information for each node.
						// BASIC_TRUCKS: store the information for each truck. Carriage
						// 进行一次bin packing
						/**
						 * 创建几个车子，长度为无限长。
						 */
						Solution_vrp packing_result = new Solution_vrp();
						// double avg_packlength = 0.0;
						for (int j = 0; j < TRUCKTYPE_NUM; j++) {
							Carriage c = new Carriage(BASIC_TRUCKS.get(j));
							c.setLength(1000000.0);
							c.setCapacity(1000000.0);
							/**
							 * 用这个车子对每个节点的boxes进行装箱。
							 */
							Route rftt = new Route();// route for this truck

							// c.setLength(BASIC_TRUCKS.get(j).getLength());
							// c.setCapacity(BASIC_TRUCKS.get(j).getLength());
							rftt.setCarriage(c);
							rftt.binpacking(curr_clients,pack_method);
							rftt.getCarriage().setCapacity(BASIC_TRUCKS.get(j).getLength());
							packing_result.getRoutes().add(rftt);
							// avg_packlength+=rftt.getCarriage().getLength();
						}
						n_3DPacking++;
						// if(isLog)
						// System.out.print(String.format("%.1f\t",
						// avg_packlength/TRUCKTYPE_NUM));
						if (isOutput) {
							SolutionSet_vrp test_out = new SolutionSet_vrp();
							test_out.add(packing_result);
							outputJSON(test_out, filenames[fileidx], PlatformIDCodeMap,
									output_directory + '/' + filenames[fileidx]);
						}
						if (isLog)
							System.out.println("finished the first binpacking!!!!");
						// System.exit(0);
						//***********************************************************************
						for (int greedy_flag = 0; greedy_flag < GREEDY_ITERATION; greedy_flag++) {// 0,1,2,3,4
							boolean[] client_checked = new boolean[CLIENT_NUM];// 是否已經轉載。
							for (int j = 0; j < CLIENT_NUM; j++) {
								client_checked[j] = false;
							}
							int nearest_node = -1;
							if (greedy_flag == 0) {// 選擇離
								double nearest_dis = Double.MAX_VALUE;
								for (int j = 0; j < CLIENT_NUM; j++) {
									client_checked[j] = false;
									String twoPlatform = String.valueOf("0+" + clients.get(j).getPlatformID());
									// System.out.println(twoPlatform);
									double dist0 = distanceMap.get(twoPlatform);
									if (dist0 < nearest_dis) {
										nearest_dis = dist0;
										nearest_node = j;
									}
								}
							} else if (greedy_flag == 1) {
								double nearest_dis = Double.MIN_VALUE;
								for (int j = 0; j < CLIENT_NUM; j++) {
									client_checked[j] = false;
									if (clients.get(j).getGoodsVolumn() > nearest_dis) {
										nearest_dis = clients.get(j).getGoodsVolumn();
										nearest_node = j;
									}
								}
							} else {
								if (PseudoRandom.randDouble() < 0.5) {
									nearest_node = PseudoRandom.randInt(0, CLIENT_NUM - 1);
								} else {
									double nearest_dis = Double.MAX_VALUE;
									for (int j = 0; j < CLIENT_NUM; j++) {
										client_checked[j] = false;
										String twoPlatform = String.valueOf("0+" + clients.get(j).getPlatformID());
										double dist0 = distanceMap.get(twoPlatform);
										if (dist0 < nearest_dis) {
											nearest_dis = dist0;
											nearest_node = j;
										}
									}
								}
							}
							/**
							 * 初始化solutionSet,对所有mustFirst节点或者第一个非mustFirst节点添加V条路径。
							 * 第一个节点必须添加。第二个节点之后如果是mustFirst则继续添加。
							 */
							boolean is_nearest_node_add = false;
							while (true) {
								int i = -1;
								for (int j = 0; j < CLIENT_NUM; j++) {
									if (!client_checked[j] && clients.get(j).isMustFirst()) {
										i = j;
										is_nearest_node_add = true;// 当有mustfirst节点时，不需要家nearest node
										break;
									}
								}
								if (i < 0) {
									// 没有mustfirst节点了。
									if (!is_nearest_node_add) {
										i = nearest_node;
										is_nearest_node_add = true;
									} else {
										break;
									}
								}
								client_checked[i] = true;
								// 找到当前要加入的节点。
								// 1.mustfirst node
								// 2.nearest_node
								boolean is_loaded = false;// 表示这个节点是否装载。
								/**
								 * 1. 先判断是否能够被一个车子装下。 1.1. 如果能够被很多的车子装下，只选择装载率最高的2种车子装。 2. 如果不能，则进行拆分装。
								 * 选择装载率高的车子。 这些箱子的重量和体积是一样的。
								 * 
								 */
								boolean[] is_checked = new boolean[TRUCKTYPE_NUM];
								for (int j = 0; j < TRUCKTYPE_NUM; j++) {// 当前解的每个车子。
									is_checked[j] = false;
								}
								int n_times = 0;
								if (isLog)
									System.out
											.println("There are " + TRUCKTYPE_NUM + "trucks need to check.");
								while (true) {
									// 找到装载率最高的车子。
									int curr_v = -1;
									double min_volumn = Double.MAX_VALUE;
									for (int v = 0; v < TRUCKTYPE_NUM; v++) {
										Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
										double length_need = nodei.getDemands();//
										// double weight_need = nodei.getWeights();
										double volumn = length_need * BASIC_TRUCKS.get(v).getWidth()
												* BASIC_TRUCKS.get(v).getHeight();
										if (volumn <= 0) {
											is_checked[v] = true;
											volumn = Double.MAX_VALUE;
										} // 这个平台的箱子不能被当前车子所容。
										if (volumn <= min_volumn && !is_checked[v]) {
											min_volumn = volumn;
											curr_v = v;
										}
									}
									if (curr_v < 0)
										break;/** 装所有可以装的车子。 **/
									int v = curr_v;
									is_checked[v] = true;

									// for(int v:selected_vehicles) {
									Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
									double length_need = nodei.getDemands();//
									double weight_need = nodei.getWeights();
									if (length_need <= 0.0)
										continue;// 有些节点不能用某种车来装。
									if (isLog)
										System.out.println(
												"判断车子:" + BASIC_TRUCKS.get(v).getTruckTypeCode() + "需求长度为： "
														+ length_need + "剩余长度为：" + BASIC_TRUCKS.get(v).getLength());

									if (length_need <= BASIC_TRUCKS.get(v).getLength()
											&& weight_need <= BASIC_TRUCKS.get(v).getCapacity()) {// 需要的长度<车子的总长。

										/**
										 * 如果一辆车子可以装下。
										 */
										Solution_vrp solution = new Solution_vrp();
										solution.distanceMap = distanceMap;
										Route route = new Route((int) (1000 + Math.random() * 8999));
										Carriage vehicle = new Carriage(BASIC_TRUCKS.get(v));
										route.setCarriage(vehicle);

										LinkedList<Node> nodes = new LinkedList<Node>();
										nodes.add(new Node(depot_start));
										nodes.add(nodei);
										nodes.add(new Node(depot_end));
										route.setNodes(nodes);
										route.setExcessLength(nodei.getDemands());// 设置已经装载的总长度。
										route.setExcessWeight(nodei.getWeights());
										if(PACKING_RECORD)
											route.getCarriage().setTruckTypeCode("pfrs"+pack_method);
										solution.getRoutes().add(route);
										solutionSet.add(solution);
										n_times++;
										is_loaded = true;
									}
								} // while(true)
								/**
								 * 如果一个车子装不下。
								 */
								if (!is_loaded) {// 如果这个车子的总长大于这个节点需要的长度。
									if (isLog)
										System.out.println("divide this client.!!!!");
									/**
									 * 如果这个节点要装到这种车子里，则必须要进行隔断。 包含多条路径的初始解。
									 */
									is_checked = new boolean[TRUCKTYPE_NUM];
									for (int j = 0; j < is_checked.length; j++) {// 当前解的每个车子。
										is_checked[j] = false;// 检查每个车子，从需要的车子数量小的开始。
									}
									n_times = 0;

									if (isLog)
										System.out.println(
												"There are " + TRUCKTYPE_NUM + " vehicles need to check.");
									while (true) {
										/** 找到需要车子最少的那种车子类型。 */
										int curr_v = -1;
										int min_n = Integer.MAX_VALUE;
										for (int v = 0; v < TRUCKTYPE_NUM; v++) {
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											double length_need = nodei.getDemands();//
											// double weight_need = nodei.getWeights();
											int n = (int) Math.ceil(length_need / BASIC_TRUCKS.get(v).getLength());
											if (n <= 0)
												is_checked[v] = true;// n = Integer.MAX_VALUE;//这个平台的箱子不能被当前车子所容。
											if (n <= min_n && !is_checked[v]) {
												min_n = n;
												curr_v = v;
											}
											// n_needed.add(n);
										}
										if (curr_v < 0 || n_times > 0)
											break;
										int v = curr_v;
										is_checked[v] = true;
										Solution_vrp solution = new Solution_vrp();
										solution.distanceMap = distanceMap;
										// ArrayList<Integer> loadBoxIdx=new ArrayList<Integer>();
										// double curr_line = 0.0;//没有加载的箱子的最前端。
										// int curr_box_idx = 0;//下一个要加载的箱子下标。
										Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
										ArrayList<Box> goods = nodei.getGoods();// 当前节点所有的boxes
										boolean is_succ = true;
										// int box_idx_before = 0;

										/**
										 * 开始装车子。 装这个车子的长度的箱子。也就是说z+length<length_limit的箱子都可以装进当前的车子里面。 检查可行性： 1.
										 * 所有装进车子里面的箱子不会支撑没在车子里面的箱子。如果有箱子支撑，则把这个箱子移动到外面。 2. 所有车子外面的箱子不会支撑里面的箱子。 3.
										 * 检查所有箱子的重量是否超出载重。
										 * 
										 * 如果支撑面积的约束和载重约束不满足，则移出去一个箱子。直到满足约束为止。
										 * 
										 * 
										 * 
										 * 以车厢长度为线作为切割点 1. 将z+length<切割线的箱子作为内箱，其他的为外箱。 2.
										 * 如果内箱的某个箱子的支撑面积约束不满足，将分割线移动到这个箱子的z坐标。 2.如果外箱的支撑面约束不满足，
										 */

										/**
										 * 里外相互不支撑，如果支撑，支撑面积应该很小<20% 如果不满足约束，那么往里面移动一个horizontal line
										 */
										boolean[] is_added = new boolean[goods.size()];
										for (int idx = 0; idx < goods.size(); idx++)
											is_added[idx] = false;
										int add_num = 0;
										double line_before = 0.0;// 车外未加载箱子的最前端。
										double line_last = 0.0;// 车内加载箱子的最后段。
										while (add_num < goods.size()) {// 表示还有箱子没有装进来。
											/**
											 * 如果还有没有load的箱子，则继续添加路径。
											 */
											Route route = new Route((int) (1000 + Math.random() * 8999));
											route.setCarriage(new Carriage(BASIC_TRUCKS.get(v)));
											
											/**
											 * 找到一个合理的分割线。
											 */
											// 记录所有的车内箱子的horizontal line
											ArrayList<Integer> inCar = new ArrayList<Integer>();// the boxes in car.
											ArrayList<Integer> outCar = new ArrayList<Integer>();// the boxes out
																									// car.
											double divide_line = line_before + BASIC_TRUCKS.get(v).getLength();// 刚开始设置的分割线。
											while (true) {
												// boolean check_support = false;//分割线没有将车外箱子分割时，不需要check_support.
												/**
												 * 根据分割线将箱子分割成车内和车外。
												 */
												inCar = new ArrayList<Integer>();// the boxes in car.
												outCar = new ArrayList<Integer>();// the boxes out car.
												double added_weight = 0.0;// 车内箱子的总重量。
												ArrayList<Double> horizontal_line = new ArrayList<Double>();
												for (int idx = 0; idx < goods.size(); idx++) {
													if (!is_added[idx]) {
														Box curr_box = goods.get(idx);
														// 最后端-最前端的长度<vehicle.length
														if (curr_box.getZCoor() >= line_before
																&& curr_box.getZCoor()
																		+ curr_box.getLength() <= divide_line
																&& Math.max(line_last,
																		curr_box.getZCoor() + curr_box.getLength())
																		- line_before <= BASIC_TRUCKS.get(v)
																				.getLength()
																&& curr_box.getWeight()
																		+ added_weight <= BASIC_TRUCKS.get(v)
																				.getCapacity()) {
															inCar.add(idx);
															if (curr_box.getZCoor()
																	+ curr_box.getLength() > line_last)
																line_last = curr_box.getZCoor()
																		+ curr_box.getLength();// 更新车内箱子的最后端。
															added_weight = added_weight + curr_box.getWeight();
															/**
															 * 将z+length加入到horizontal_line中，并且按照降序排列。
															 */
															horizontal_line.add(
																	curr_box.getZCoor() + curr_box.getLength());
														} else {
															outCar.add(idx);
															if (curr_box.getZCoor() < divide_line
																	&& curr_box.getZCoor()
																			+ curr_box.getLength() > divide_line) {
																// check_support = true;
															}
														}
													}
												}
												// Comparator.naturalOrder()升序。
												// Comparator.reverseOrder()降序。
												horizontal_line.sort(Comparator.reverseOrder());
												// for(int idx = horizontal_line.size()-1;idx>0;idx--) {
												/**
												 * 判断内外boxes相互不支撑。
												 */
												boolean support = false;
												// if(check_support) {
												/**
												 * 判断里面的箱子是否被外面的箱子支配。
												 */
												for (int in : inCar) {
													/**
													 * //计算该箱子的底部面积。 计算currBox被支撑的面积。
													 */
													Box currBox = goods.get(in);
													double bottomArea = currBox.getWidth() * currBox.getLength();
													double curr_y = currBox.getYCoor();
													double crossArea = 0;
													support = false;
													for (int out : outCar) {
														/**
														 * 判断车外箱子是否支撑车内箱子。
														 */
														// Box b_in = goods.get(in);
														// Box b_out = goods.get(out);
														Box existBox = goods.get(out);
														if (Math.abs(existBox.getYCoor() + existBox.getHeight()
																- curr_y) <= 0.0) {
															double xc = currBox.getXCoor(), zc = currBox.getZCoor(),
																	xe = existBox.getXCoor(),
																	ze = existBox.getZCoor();
															double wc = currBox.getWidth(),
																	lc = currBox.getLength(),
																	we = existBox.getWidth(),
																	le = existBox.getLength();

															if (!((xc + wc < xe) || (xe + we < xc) || (zc + lc < ze)
																	|| (ze + le < zc))) {
																double[] XCoor = { xc, xc + wc, xe, xe + we };
																double[] ZCoor = { zc, zc + lc, ze, ze + le };
																// sort xc,xc+wc,xe,xe+we
																Arrays.sort(XCoor);
																Arrays.sort(ZCoor);
																// sort zc,zc+lc,ze,ze+le
																crossArea = crossArea
																		+ Math.abs(XCoor[2] - XCoor[1])
																				* Math.abs(ZCoor[2] - ZCoor[1]);
																if (crossArea > 0.01 * bottomArea) {
																	/**
																	 * 如果有支撑面积。
																	 */
																	support = true;
																	break;
																}
															}
														}
													}
													if (support)
														break;
												}
												if (!support) {
													/**
													 * 判断外面的箱子是否被里面的箱子支配。
													 */
													for (int in : outCar) {
														/**
														 * //计算该箱子的底部面积。 计算currBox被支撑的面积。
														 */
														Box currBox = goods.get(in);
														// double bottomArea =
														// currBox.getWidth()*currBox.getLength();
														double curr_y = currBox.getYCoor();
														double crossArea = 0;
														support = false;
														for (int out : inCar) {
															/**
															 * 判断车外箱子是否支撑车内箱子。
															 */
															// Box b_in = goods.get(in);
															// Box b_out = goods.get(out);
															Box existBox = goods.get(out);
															if (Math.abs(existBox.getYCoor() + existBox.getHeight()
																	- curr_y) <= 0.0) {
																double xc = currBox.getXCoor(),
																		zc = currBox.getZCoor(),
																		xe = existBox.getXCoor(),
																		ze = existBox.getZCoor();
																double wc = currBox.getWidth(),
																		lc = currBox.getLength(),
																		we = existBox.getWidth(),
																		le = existBox.getLength();

																if (!((xc + wc < xe) || (xe + we < xc)
																		|| (zc + lc < ze) || (ze + le < zc))) {
																	double[] XCoor = { xc, xc + wc, xe, xe + we };
																	double[] ZCoor = { zc, zc + lc, ze, ze + le };
																	// sort xc,xc+wc,xe,xe+we
																	Arrays.sort(XCoor);
																	Arrays.sort(ZCoor);
																	// sort zc,zc+lc,ze,ze+le
																	crossArea = crossArea
																			+ Math.abs(XCoor[2] - XCoor[1])
																					* Math.abs(ZCoor[2] - ZCoor[1]);
																	if (crossArea > 0) {
																		/**
																		 * 如果有支撑面积。
																		 */
																		support = true;
																		break;
																	}
																}
															}
														}
														if (support)
															break;
													}
												} // }
												if (!support&&inCar.size()>0) {
													/**
													 * 找到了可行的分割线。
													 */
													// divide_line
													break;
												} else {
													// 这条分割线不合理。那么找下一个分割线。
													int idx = 0;
													while (idx < horizontal_line.size()
															&& horizontal_line.get(idx) >= divide_line)
														idx++;// horizontal_line是从大到小排列的。
													if (idx < horizontal_line.size())
														divide_line = horizontal_line.get(idx);
													else {
														if (isLog)
															System.out.println("不能分割。");
														is_succ = false;
														break;
													}
													// break;//
												}
												// }//for each horizontal line
											}
											/**
											 * 判断分割线是否合理。 1. 车内total box weight < capacity. 2.
											 * 里外相互不支撑！！如果支撑，则其总的支撑面-里外支撑面>80% 如果不满足约束，则将分割线往里面移动一下。
											 */

											if (!is_succ) {
												break;
											}
											ArrayList<Box> boxes_node0 = new ArrayList<Box>();

											double weight_temp = 0.0;
											double minz_load_box = Double.MAX_VALUE; // 相当于
											double maxz_load_box = Double.MIN_VALUE;
											for (int load : inCar) {
												if (goods.get(load).getZCoor() < minz_load_box) {
													minz_load_box = goods.get(load).getZCoor();
												}
												if (goods.get(load).getZCoor()
														+ goods.get(load).getLength() > maxz_load_box)
													maxz_load_box = goods.get(load).getZCoor()
															+ goods.get(load).getLength();
											}
											for (int load : inCar) {
												Box load_box = new Box(goods.get(load));
												load_box.setZCoor(goods.get(load).getZCoor() - minz_load_box);
												boxes_node0.add(load_box);
												weight_temp = weight_temp + load_box.getWeight();
												add_num = add_num + 1;
												is_added[load] = true;
											}
											/**
											 * 计算未加载箱子的最前端。
											 */
											line_before = Double.MAX_VALUE;
											for (int unload : outCar) {
												if (goods.get(unload).getZCoor() < line_before)
													line_before = goods.get(unload).getZCoor();
											}
											route.setBoxes(boxes_node0);// 这里先不设置boxes，注意在输出解的时候，我们要转换到route的boxes里面。
											Node divide_node = new Node(
													packing_result.getRoutes().get(v).getNodes().get(i));
											divide_node.setGoods(boxes_node0);
											divide_node.setDemands(maxz_load_box - minz_load_box);//
											divide_node.setWeights(weight_temp);

											LinkedList<Node> nodes = new LinkedList<Node>();
											nodes.add(new Node(depot_start));
											nodes.add(divide_node);
											nodes.add(new Node(depot_end));
											route.setNodes(nodes);
											route.setExcessLength(divide_node.getDemands());// 设置已经装载的总长度。
											route.setExcessWeight(divide_node.getWeights());
											if(PACKING_RECORD)
												route.getCarriage().setTruckTypeCode("pfrs"+pack_method);
											solution.getRoutes().add(route);

										}
										if (is_succ) {
											solutionSet.add(solution);
											n_times++;
										}
									}
								} // 遍历每辆车子
							} // 遍历每个节点。
							if (isOutput)
								output(solutionSet, filenames[fileidx], PlatformIDCodeMap);
							if (isLog)
								System.out.println("finished add the first node.");

							/**
							 * 对于每一个叶子节点的解(当前solutionSet里面的解）进行增加节点。
							 */
							SolutionSet_vrp solutionSet_offspring = new SolutionSet_vrp();
							while (true) {
								int min_i = -1;
								if (greedy_flag == 0) {
									// 选择与已经加载的节点最近的节点进行加入。
									double min_dist = Double.MAX_VALUE;
									for (int ii = 0; ii < CLIENT_NUM; ii++) {
										if (!client_checked[ii]) {
											for (int jj = 0; jj < CLIENT_NUM; jj++) {
												if (client_checked[jj]) {// 没有加载的ii到已经加载的节点jj的距离。
													if (!clients.get(jj).isMustFirst()) {
														// 判断前面。
														String twoPlatform = String.valueOf(String
																.valueOf(clients.get(ii).getPlatformID()) + "+"
																+ String.valueOf(clients.get(jj).getPlatformID()));
														// System.out.println(twoPlatform);
														double dist0 = distanceMap.get(twoPlatform);
														if (dist0 < min_dist) {
															min_dist = dist0;
															min_i = ii;
														}
													}
													// 判断后面。。
													String twoPlatform = String.valueOf(String
															.valueOf(clients.get(jj).getPlatformID()) + "+"
															+ String.valueOf(clients.get(ii).getPlatformID()));
													// System.out.println(twoPlatform);
													double dist0 = distanceMap.get(twoPlatform);
													if (dist0 < min_dist) {
														min_dist = dist0;
														min_i = ii;
													}
												}
											}
										}
									}
								} else if (greedy_flag == 1) {
									// 选择体积最大的。先体积大的，后体积小的。
									double max_volumn = Double.MIN_VALUE;
									for (int ii = 0; ii < CLIENT_NUM; ii++) {
										if (!client_checked[ii]) {
											if (clients.get(ii).getGoodsVolumn() > max_volumn) {// 挑选剩下的里面volumn最高的。
												max_volumn = clients.get(ii).getGoodsVolumn();
												min_i = ii;
											}
										}
									}
								} else {
									if (PseudoRandom.randDouble() < 0.2) {
										// 随机选择一个。
										int[] permutation = new int[CLIENT_NUM];
										randomPermutation(permutation, CLIENT_NUM);
										for (int ii = 0; ii < CLIENT_NUM; ii++) {
											if (!client_checked[permutation[ii]]) {
												min_i = permutation[ii];
												break;
											}
										}
									} else {
										// 选择与已经加载的节点最近的节点进行加入。
										double min_dist = Double.MAX_VALUE;

										for (int ii = 0; ii < CLIENT_NUM; ii++) {
											if (!client_checked[ii]) {
												for (int jj = 0; jj < CLIENT_NUM; jj++) {
													if (client_checked[jj]) {// 没有加载的ii到已经加载的节点jj的距离。
														if (!clients.get(jj).isMustFirst()) {
															// 判断前面。
															String twoPlatform = String.valueOf(String
																	.valueOf(clients.get(ii).getPlatformID()) + "+"
																	+ String.valueOf(
																			clients.get(jj).getPlatformID()));
															// System.out.println(twoPlatform);
															double dist0 = distanceMap.get(twoPlatform);
															if (dist0 < min_dist) {
																min_dist = dist0;
																min_i = ii;
															}
														}
														// 判断后面。。
														String twoPlatform = String.valueOf(String
																.valueOf(clients.get(jj).getPlatformID()) + "+"
																+ String.valueOf(clients.get(ii).getPlatformID()));
														// System.out.println(twoPlatform);
														double dist0 = distanceMap.get(twoPlatform);
														if (dist0 < min_dist) {
															min_dist = dist0;
															min_i = ii;
														}
													}
												}
											}
										}
									}
								}
								if (min_i < 0)
									break;
								int i = min_i;
								client_checked[i] = true;
								if (isLog)
									System.out.println("============================begin platform: "
											+ PlatformIDCodeMap.get(clients.get(i).getPlatformID())
											+ "======================================");
								for (int sidx = 0; sidx < solutionSet.solutionList_.size(); sidx++) {
									Solution_vrp curr_solution = solutionSet.solutionList_.get(sidx);
									if (isLog)
										System.out.println("=================begin " + sidx
												+ "th solution========================");
									/**
									 * 对于当前解，如果现有的车子可以装下，则装入现有的车子。 如果没有现有的车子可以装下，则新增V个车子的解来装。也就是说从当前解->裂变出V个解。
									 */
									/**
									 * 先判断现有的车子是否可以装下。
									 */
									boolean is_loaded = false;

									if (isLog)
										System.out.println("*step1: insert to 1 exist route.");
									/**
									 * 1. 选择最适合的路径进行插入。避免尝试太多的解。 最适合：剩余空间最小的。 最适合：距离增加最少的。
									 */
									boolean[] is_checked = new boolean[TRUCKTYPE_NUM];
									for (int j = 0; j < TRUCKTYPE_NUM; j++) {// 当前解的每个车子。
										is_checked[j] = false;
									}
									int n_times = 0;
									double checked_min_volumn = 0.0;
									while (true) {
										// 1. 选择一辆车。
										int curr_v = -1;
										double min_volumn = Double.MAX_VALUE;
										for (int v = 0; v < TRUCKTYPE_NUM; v++) {
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											double length_need = nodei.getDemands();//
											// double weight_need = nodei.getWeights();
											double volumn = length_need * BASIC_TRUCKS.get(v).getWidth()
													* BASIC_TRUCKS.get(v).getHeight();
											if (volumn <= 0) {
												is_checked[v] = true;
												volumn = Double.MAX_VALUE;
											} // 这个平台的箱子不能被当前车子所容。
											if (volumn > checked_min_volumn && volumn <= min_volumn
													&& !is_checked[v]) {
												min_volumn = volumn;
												curr_v = v;
											}
											// all_volumn.add(volumn);
										}
										if (curr_v < 0 || n_times > N_TIME_STEP1)
											break;
										checked_min_volumn = min_volumn;
										int v = curr_v;
										is_checked[v] = true;

										/** 得到这种车型模式下要插入的节点。 */
										Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
										/*** 这个节点的总长度是多少？ */
										double length_need = nodei.getDemands();//
										double weight_need = nodei.getWeights();
										if (length_need <= 0.0)
											continue;// 有些节点不能用某种车来装。

										/**
										 * 选择一条新增距离最短的路径来装载。
										 */
										ArrayList<L> LSequence = new ArrayList<L>();

										for (int j = 0; j < curr_solution.getRoutes().size(); j++) {// 如果加入，新增距离时多少？
											Route exist_routej = curr_solution.getRoutes().get(j);// 现有的第j条路径。
											double residual_z = exist_routej.getCarriage().getLength()
													- exist_routej.getExcessLength();// 这个车子还有多长可以装的？
											double residual_w = exist_routej.getCarriage().getCapacity()
													- exist_routej.getExcessWeight();

											boolean height_ok = (exist_routej.getCarriage()
													.getHeight() >= BASIC_TRUCKS.get(v).getHeight());
											boolean width_ok = (exist_routej.getCarriage()
													.getWidth() >= BASIC_TRUCKS.get(v).getWidth());
											if (height_ok && width_ok && length_need <= residual_z
													&& weight_need <= residual_w) {
												/** 如果现有的车子j可以装载当前模式节点，则记录新增距离。 */
												// if(isLog)
												// System.out.println("operate on "+j+"with "+
												// exist_routej.getNodes().size()+" nodes.");
												// 加入到当前这个现有的车子里面中，找一个距离最小的地方插入。
												double min_increase = Double.MAX_VALUE;
												int insert_position = -1;// 插入到这个节点的后面。
												// 遍历当前可行路径的所有节点
												for (int k = 0; k < exist_routej.getNodes().size() - 1; k++) {
													if (exist_routej.getNodes().get(k + 1).isMustFirst())
														continue;
													/**
													 * 原来是0-1-2，则可以插入0或者1的后面。 原来是0-1-2-3，则可以插入0，1，2，的后面。
													 * 若是插入0后面，则0-new-1-2-3，则距离增加0-new-1减去0-1 找到距离增加最少的位置。
													 */
													String twoPlatform = String.valueOf(exist_routej.getNodes()
															.get(k).getPlatformID() + "+"
															+ exist_routej.getNodes().get(k + 1).getPlatformID());
													// System.out.println(twoPlatform);
													double dist_before = distanceMap.get(twoPlatform);
													twoPlatform = String
															.valueOf(exist_routej.getNodes().get(k).getPlatformID()
																	+ "+" + clients.get(i).getPlatformID());
													double dist_after = distanceMap.get(twoPlatform);
													twoPlatform = String.valueOf(clients.get(i).getPlatformID()
															+ "+"
															+ exist_routej.getNodes().get(k + 1).getPlatformID());
													dist_after = dist_after + distanceMap.get(twoPlatform);
													if (dist_after - dist_before < min_increase) {
														min_increase = dist_after - dist_before;
														insert_position = k;
													}
												} // 遍历当前可行路径的所有节点
													// 记录加入到j路径的位置insert_position，新增距离是多少。
												L l = new L();
												l.setI(j);
												l.setJ(insert_position);
												l.setSij(min_increase);
												LSequence.add(l);
											} // if basic constraint is ok
											else {
												/**
												 * 这辆车不够空间来转载当前节点。
												 */
											} //
										} // for 遍历完了当前解的所有路径（车子）。

										/** 找到距离增加最少的路径来加入。 **/
										Collections.sort(LSequence);
										for (int il = 0; il < LSequence.size(); il++) {
											/**
											 * 对这个solution进行改变。 将当前节点插入到exist_routej的insert_position节点后面。
											 */
											int j = LSequence.get(il).getI();
											Route exist_routej = curr_solution.getRoutes().get(j);
											int insert_position = LSequence.get(il).getJ();
											// 改变routej
											Solution_vrp new_solution = new Solution_vrp(curr_solution);
											Route new_routej = new_solution.getRoutes().get(j);
											LinkedList<Node> ns = new LinkedList<Node>();

											for (int k = 0; k < exist_routej.getNodes().size(); k++) {// 遍历当前可行路径的所有节点
												ns.add(new Node(exist_routej.getNodes().get(k)));
												if (k == insert_position) {
													ns.add(new Node(nodei));
												}
											}

											new_routej.setNodes(ns);
											new_routej.setExcessLength(
													exist_routej.getExcessLength() + nodei.getDemands());// 已经装载的长度。
											new_routej.setExcessWeight(
													exist_routej.getExcessWeight() + nodei.getWeights());// 已经装载的重量。
											// new_solution.getRoutes().set(j, new_routej);//替换routej
											solutionSet_offspring.add(new_solution);
											n_times++;// 表示成功产生一个新解。

											if (isLog)
												System.out.println("Add one solution and finished current parent.");
											is_loaded = true;
											break;
										} // for LSequence
									} // while true

									/**
									 * 2. 如果加入一条路径不行，那就加入两条路径呗。
									 */
									if (!is_loaded) {
										if (isLog)
											System.out.println("*step2: insert to 2 route.");

										is_checked = new boolean[TRUCKTYPE_NUM];
										for (int j = 0; j < TRUCKTYPE_NUM; j++) {// 当前解的每个车子。
											is_checked[j] = false;
										}
										n_times = 0;
										checked_min_volumn = 0.0;
										while (true) {
											int curr_v = -1;
											double min_volumn = Double.MAX_VALUE;
											for (int v = 0; v < TRUCKTYPE_NUM; v++) {
												Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
												double length_need = nodei.getDemands();//
												// double weight_need = nodei.getWeights();
												double volumn = length_need * BASIC_TRUCKS.get(v).getWidth()
														* BASIC_TRUCKS.get(v).getHeight();
												if (volumn <= 0) {
													is_checked[v] = true;
													volumn = Double.MAX_VALUE;
												} // 这个平台的箱子不能被当前车子所容。
												if (volumn > checked_min_volumn && volumn <= min_volumn
														&& !is_checked[v]) {
													min_volumn = volumn;
													curr_v = v;
												}
												// all_volumn.add(volumn);
											}
											if (curr_v < 0 || n_times > N_TIME_STEP2)
												break;// 所有可能的组合。
											checked_min_volumn = min_volumn;
											int v = curr_v;
											is_checked[v] = true;
											// for(int v=0;v<TRUCKTYPE_NUM;v++) {//对每种模式都进行判断。
											// n_times=0;
											// 当前节点需要的长度和重量。
											/** 得到要插入的节点。 */
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											/*** 这个节点的总长度是多少？ */
											double length_need = nodei.getDemands();//
											double weight_need = nodei.getWeights();
											ArrayList<Box> goods = nodei.getGoods();// 当前节点所有的boxes，这些boxes已经有坐标了。
											if (length_need <= 0.0)
												continue;// 有些节点不能用某种车来装。

											ArrayList<L> LSequence = new ArrayList<L>();
											for (int route_i = 0; route_i < curr_solution.getRoutes()
													.size(); route_i++) {// 当前解的每个车子。
												Route exist_routei = curr_solution.getRoutes().get(route_i);
												for (int route_j = 0; route_j < curr_solution.getRoutes()
														.size(); route_j++) {// 当前解的每个车子。
													if (route_i == route_j)
														continue;
													Route exist_routej = curr_solution.getRoutes().get(route_j);
													// 选择两条路径。
													// 找到位置进行插入route_i。
													double min_increasei = Double.MAX_VALUE;
													int insert_positioni = -1;// 插入到这个节点的后面。
													for (int k = 0; k < exist_routei.getNodes().size() - 1; k++) {
														if (exist_routei.getNodes().get(k + 1).isMustFirst())
															continue;
														String twoPlatform = String.valueOf(
																exist_routei.getNodes().get(k).getPlatformID() + "+"
																		+ exist_routei.getNodes().get(k + 1)
																				.getPlatformID());
														// System.out.println(twoPlatform);
														double dist_before = distanceMap.get(twoPlatform);
														twoPlatform = String.valueOf(
																exist_routei.getNodes().get(k).getPlatformID() + "+"
																		+ nodei.getPlatformID());
														double dist_after = distanceMap.get(twoPlatform);
														twoPlatform = String
																.valueOf(nodei.getPlatformID() + "+" + exist_routei
																		.getNodes().get(k + 1).getPlatformID());
														dist_after = dist_after + distanceMap.get(twoPlatform);
														if (dist_after - dist_before < min_increasei) {
															min_increasei = dist_after - dist_before;
															insert_positioni = k;
														}
													}

													// 找到位置进行插入。
													double min_increasej = Double.MAX_VALUE;
													int insert_positionj = -1;// 插入到这个节点的后面。
													for (int k = 0; k < exist_routej.getNodes().size() - 1; k++) {
														if (exist_routej.getNodes().get(k + 1).isMustFirst())
															continue;
														String twoPlatform = String.valueOf(
																exist_routej.getNodes().get(k).getPlatformID() + "+"
																		+ exist_routej.getNodes().get(k + 1)
																				.getPlatformID());
														// System.out.println(twoPlatform);
														double dist_before = distanceMap.get(twoPlatform);
														twoPlatform = String.valueOf(
																exist_routej.getNodes().get(k).getPlatformID() + "+"
																		+ nodei.getPlatformID());
														double dist_after = distanceMap.get(twoPlatform);
														twoPlatform = String
																.valueOf(nodei.getPlatformID() + "+" + exist_routej
																		.getNodes().get(k + 1).getPlatformID());
														dist_after = dist_after + distanceMap.get(twoPlatform);
														if (dist_after - dist_before < min_increasej) {
															min_increasej = dist_after - dist_before;
															insert_positionj = k;
														}
													}
													L l = new L();
													l.setI(route_i);
													l.setJ(route_j);
													l.setM(insert_positioni);
													l.setN(insert_positionj);
													l.setSij(min_increasei + min_increasej);
													LSequence.add(l);
												} // for route j
											} // for route i

											Collections.sort(LSequence);
											for (int il = 0; il < LSequence.size(); il++) {
												int route_i = LSequence.get(il).getI();
												int route_j = LSequence.get(il).getJ();
												int insert_positioni = LSequence.get(il).getM();
												int insert_positionj = LSequence.get(il).getN();
												Route exist_routei = curr_solution.getRoutes().get(route_i);
												Route exist_routej = curr_solution.getRoutes().get(route_j);
												// 看两条路径的剩余长度之和是否>所需要的长度。
												// 剩余的重量是否>需要的重量。
												double residual_zi = exist_routei.getCarriage().getLength()
														- exist_routei.getExcessLength();// 这个车子还有多长可以装的？
												double residual_zj = exist_routej.getCarriage().getLength()
														- exist_routej.getExcessLength();// 这个车子还有多长可以装的？
												double residual_wi = exist_routei.getCarriage().getCapacity()
														- exist_routei.getExcessWeight();
												double residual_wj = exist_routej.getCarriage().getCapacity()
														- exist_routej.getExcessWeight();

												boolean height_ok = (exist_routei.getCarriage()
														.getHeight() >= BASIC_TRUCKS.get(v).getHeight())
														&& (exist_routej.getCarriage().getHeight() >= BASIC_TRUCKS
																.get(v).getHeight());
												boolean width_ok = (exist_routei.getCarriage()
														.getWidth() >= BASIC_TRUCKS.get(v).getWidth())
														&& (exist_routej.getCarriage().getWidth() >= BASIC_TRUCKS
																.get(v).getWidth());
												/*
												 * 判断能否装下： 1. 这两个车子的长宽高是否比当前模式小。 2. 剩余载重是否比当前模式载重小。 3.
												 */
												if (height_ok && width_ok
														&& length_need <= residual_zi + residual_zj
														&& weight_need <= residual_wi + residual_wj) {
													/*** 进行切割，并装载到两条路径里面。 */
													// 1. 先满足routei再满足给routej，residual_zi+residual_wi
													// 1.1 找到一个合理的分割线。
													double[] length_bound = { residual_zi, residual_zj };
													double[] weight_bound = { residual_wi, residual_wj };
													boolean[] is_succ = divide_node(goods, length_bound,
															weight_bound);
													if (is_succ != null) {
														/** 如果成功了。则添加一个新的子代解。 **/
														Solution_vrp new_solution = new Solution_vrp(curr_solution);
														// 拆分成两个node分别加入到两条路径里面。
														ArrayList<Box> boxes_in = new ArrayList<Box>();
														ArrayList<Box> boxes_out = new ArrayList<Box>();
														double weight_in = 0.0, weight_out = 0.0;
														double minz_in = Double.MAX_VALUE,
																minz_out = Double.MAX_VALUE;
														double maxz_in = Double.MIN_VALUE,
																maxz_out = Double.MIN_VALUE;
														for (int goodsi = 0; goodsi < goods.size(); goodsi++) {
															if (!is_succ[goodsi]) {
																// in car
																if (goods.get(goodsi).getZCoor() < minz_in) {
																	minz_in = goods.get(goodsi).getZCoor();
																}
																if (goods.get(goodsi).getZCoor()
																		+ goods.get(goodsi).getLength() > maxz_in) {
																	maxz_in = goods.get(goodsi).getZCoor()
																			+ goods.get(goodsi).getLength();
																}
															} else {
																// out car
																if (goods.get(goodsi).getZCoor() < minz_out) {
																	minz_out = goods.get(goodsi).getZCoor();
																}
																if (goods.get(goodsi).getZCoor() + goods.get(goodsi)
																		.getLength() > maxz_out) {
																	maxz_out = goods.get(goodsi).getZCoor()
																			+ goods.get(goodsi).getLength();
																}
															}
														}
														for (int goodsi = 0; goodsi < goods.size(); goodsi++) {
															if (!is_succ[goodsi]) {
																Box load_box = new Box(goods.get(goodsi));
																load_box.setZCoor(
																		goods.get(goodsi).getZCoor() - minz_in);
																boxes_in.add(load_box);
																weight_in = weight_in + load_box.getWeight();
															} else {
																Box load_box = new Box(goods.get(goodsi));
																load_box.setZCoor(
																		goods.get(goodsi).getZCoor() - minz_out);
																boxes_out.add(load_box);
																weight_out = weight_out + load_box.getWeight();
															}
														}
														/** add to route i **/
														Node divide_nodei = new Node(nodei);
														divide_nodei.setGoods(boxes_in);
														divide_nodei.setDemands(maxz_in - minz_in);//
														divide_nodei.setWeights(weight_in);

														Route new_routei = new_solution.getRoutes().get(route_i);// new
																													// Route(exist_routei);
														LinkedList<Node> nsi = new LinkedList<Node>();
														for (int k = 0; k < exist_routei.getNodes().size(); k++) {
															nsi.add(new Node(exist_routei.getNodes().get(k)));
															if (k == insert_positioni) {
																nsi.add(new Node(divide_nodei));
															}
														}
														new_routei.setNodes(nsi);
														new_routei.setExcessLength(new_routei.getExcessLength()
																+ divide_nodei.getDemands());
														new_routei.setExcessWeight(new_routei.getExcessWeight()
																+ divide_nodei.getWeights());
														/** add to route j **/
														Node divide_nodej = new Node(nodei);
														divide_nodej.setGoods(boxes_out);
														divide_nodej.setDemands(maxz_out - minz_out);//
														divide_nodej.setWeights(weight_out);

														Route new_routej = new_solution.getRoutes().get(route_j);// new
																													// Route(exist_routej);
														LinkedList<Node> nsj = new LinkedList<Node>();
														for (int k = 0; k < exist_routej.getNodes().size(); k++) {
															nsj.add(new Node(exist_routej.getNodes().get(k)));
															if (k == insert_positionj) {
																nsj.add(new Node(divide_nodej));
															}
														}
														new_routej.setNodes(nsj);
														new_routej.setExcessLength(new_routej.getExcessLength()
																+ divide_nodej.getDemands());
														new_routej.setExcessWeight(new_routej.getExcessWeight()
																+ divide_nodej.getWeights());
														solutionSet_offspring.add(new_solution);
														n_times++;
														is_loaded = true;
														// continue;//不再进行判断了。
														break;
													} // is_succ!=null

												} // if satisfy constraints
											} // for LSequence

										} // for truck v
										if (isLog)
											System.out.println("divided to exist routes by adding " + n_times
													+ "new solutions!!!!!!");
									} // is loaded

									/**
									 * 如果不能插入到现有的路径里面去。 则尝试添加新的路径（车子）
									 */
									n_times = 0;
									if (!is_loaded) {// 为这个解创造新的路径。
										if (isLog)
											System.out.println("step3: insert to 1 new route.");
										/**
										 * 如果没有现有解可以装载当前的节点。 那么从当前解裂变出V个新解 1. 需要的长度<车子的总长。 2.
										 */
										/**
										 * 选择装载率最高的车子。
										 */
										/**
										 * 选择装载率高的车子。 这些箱子的重量和体积是一样的。
										 */
										ArrayList<Integer> selected_vehicles = new ArrayList<Integer>();
										ArrayList<Double> all_volumn = new ArrayList<Double>();
										double min_volumn = Double.MAX_VALUE;
										for (int v = 0; v < TRUCKTYPE_NUM; v++) {
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											double length_need = nodei.getDemands();//
											// double weight_need = nodei.getWeights();
											double volumn = length_need * BASIC_TRUCKS.get(v).getWidth()
													* BASIC_TRUCKS.get(v).getHeight();
											if (volumn <= 0)
												volumn = Double.MAX_VALUE;// 这个平台的箱子不能被当前车子所容。
											if (volumn <= min_volumn)
												min_volumn = volumn;
											all_volumn.add(volumn);
										}
										for (int v = 0; v < TRUCKTYPE_NUM; v++) {
											if (all_volumn.get(v) <= min_volumn) {
												selected_vehicles.add(v);
											}
										}
										// double min_volumn = Double.MAX_VALUE;
										// int select_v = 0;
										// for(int v=0;v<TRUCKTYPE_NUM;v++) {
										for (int v : selected_vehicles) {
											// Node nodei = new
											// Node(packing_result.getRoutes().get(v).getNodes().get(i));
											// double length_need = nodei.getDemands();//
											// double weight_need = nodei.getWeights();//
											// double load_ratio =
											// (BASIC_TRUCKS.get(v).getWidth()*BASIC_TRUCKS.get(v).getHeight()*length_need);//越小约好，总体的体积是一样的。
											// if(load_ratio<min_volumn) {min_volumn=load_ratio;select_v = v;}
											//
											// }
											// int v=select_v;
											if (n_times > N_TIME_STEP3)
												break;
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											double length_need = nodei.getDemands();//
											double weight_need = nodei.getWeights();//
											if (length_need <= 0.0)
												continue;// 有些节点不能用某种车来装。
											if (isLog) {
												System.out.println("判断车子:" + BASIC_TRUCKS.get(v).getTruckTypeCode()
														+ "需求载重为： " + weight_need + "剩余载重为："
														+ BASIC_TRUCKS.get(v).getCapacity());
												System.out.println("判断车子:" + BASIC_TRUCKS.get(v).getTruckTypeCode()
														+ "需求长度为： " + length_need + "剩余长度为："
														+ BASIC_TRUCKS.get(v).getLength());
											}
											if (length_need <= BASIC_TRUCKS.get(v).getLength()
													&& weight_need <= BASIC_TRUCKS.get(v).getCapacity()) {// 需要的长度<车子的总长。
												Solution_vrp solution = new Solution_vrp(curr_solution);
												// solution.distanceMap = distanceMap;
												Route route = new Route((int) (1000 + Math.random() * 8999));
												Carriage vehicle = new Carriage(BASIC_TRUCKS.get(v));
												route.setCarriage(vehicle);

												/**
												 * 处理boxes
												 * 
												 */
												// ArrayList<Box> boxes_node0 = new ArrayList<Box>();
												/**
												 * 如果前面能够记录从第几个boxes到第几个box是node i的就好了。
												 * packing_result.getNode.get(0)
												 */
												// for(Box
												// b:packing_result.getRoutes().get(v).getNodes().get(i).getGoods())
												// if(b.getPlatformid()==clients.get(0).getPlatformID())
												// boxes_node0.add(new Box(b));
												// else
												// System.out.println("this is impossbile.!!");
												// route.setBoxes(boxes_node0);//这里先不设置boxes，注意在输出解的时候，我们要转换到route的boxes里面。
												// packing_result.getRoutes().get(v).getNodes().get(i).getGoods()

												// nodei.setGoods(boxes_node0);
												LinkedList<Node> nodes = new LinkedList<Node>();
												nodes.add(new Node(depot_start));
												nodes.add(nodei);
												nodes.add(new Node(depot_end));
												route.setNodes(nodes);
												// 设置这条新路径的长度以及载重。
												route.setExcessLength(nodei.getDemands());
												route.setExcessWeight(nodei.getWeights());
												if(PACKING_RECORD)
													route.getCarriage().setTruckTypeCode("pfrs"+pack_method);
												solution.getRoutes().add(route);
												// solutionSet.add(solution);
												solutionSet_offspring.add(solution);
												// break;//找到了就结束。
												is_loaded = true;
												n_times++;
											}
										}
									} // if is_loaded

									/**
									 * 添加一个新车，剩下的加入到现有的路径里面。
									 */

									if (!is_loaded) {
										if (isLog)
											System.out.println("step3: 1 new route and 1 old route.");
									}

									/**
									 * 如果前两次都不行。则进行切割。
									 */
									if (!is_loaded) {
										if (isLog)
											System.out.println("step4: insert to n new route.");
										// if(isLog)
										// System.out.println("divide this client.!!!!");
										/**
										 * 如果1.插入不可行，2.创建新的解也不可行，那么就要进行切割。 如果需要的长度大于车子的总长的话，需要分割。
										 */
										// 如果这个车子的总长大于这个节点需要的长度。
										/**
										 * 如果这个节点要装到这种车子里，则必须要进行隔断。
										 */
										/**
										 * 选择最合适的车子进行装载。 最合适：需要的车子最少的哪个。
										 */
										is_checked = new boolean[TRUCKTYPE_NUM];
										for (int j = 0; j < is_checked.length; j++) {// 当前解的每个车子。
											is_checked[j] = false;// 检查每个车子，从需要的车子数量小的开始。
										}
										n_times = 0;
										if (isLog)
											System.out.println("There are " + TRUCKTYPE_NUM
													+ " vehicles need to check.");
										while (true) {
											/**
											 * 找到需要车子最少的那种车子类型。
											 */
											int curr_v = -1;// 当前还没加载当前节点的路径。还没用这个路径来产生新解。
											// ArrayList<Integer> selected_vehicles = new ArrayList<Integer>();
											// ArrayList<Integer> n_needed = new ArrayList<Integer>();//记录需要车子的数量。
											int min_n = Integer.MAX_VALUE;
											for (int v = 0; v < TRUCKTYPE_NUM; v++) {
												Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
												double length_need = nodei.getDemands();//
												// double weight_need = nodei.getWeights();
												int n = (int) Math
														.ceil(length_need / BASIC_TRUCKS.get(v).getLength());
												if (n <= 0)
													is_checked[v] = true;// n =
																			// Integer.MAX_VALUE;//这个平台的箱子不能被当前车子所容。
												if (n <= min_n && !is_checked[v]) {
													min_n = n;
													curr_v = v;
												}
												// n_needed.add(n);
											}
											// for(int v=0;v<TRUCKTYPE_NUM;v++) {
											// if(n_needed.get(v)<=min_n) {
											// selected_vehicles.add(v);
											// }
											// }
											if (curr_v < 0 || n_times > N_TIME_STEP4)
												break;
											int v = curr_v;
											is_checked[v] = true;// 表示已经遍历过了。
											// for(int v:selected_vehicles) {
											Solution_vrp solution = new Solution_vrp(curr_solution);
											solution.distanceMap = distanceMap;
											// ArrayList<Integer> loadBoxIdx=new ArrayList<Integer>();
											// double curr_line = 0.0;//已经加载到的地方。
											// int curr_box_idx = 0;//下一个要加载的箱子下标。
											Node nodei = packing_result.getRoutes().get(v).getNodes().get(i);
											ArrayList<Box> goods = nodei.getGoods();// 当前节点所有的boxes，这些boxes已经有坐标了。

											boolean is_succ = true;
											// int box_idx_before = 0;
											boolean[] is_added = new boolean[goods.size()];
											for (int idx = 0; idx < goods.size(); idx++)
												is_added[idx] = false;
											int add_num = 0;
											double line_before = 0.0;// 车外未加载箱子的最前端。
											double line_last = 0.0;// 车内加载箱子的最后段。
											while (add_num < goods.size()) {// 表示还有箱子没有装进来。
												/**
												 * 如果还有没有load的箱子，则继续添加路径。
												 */
												Route route = new Route((int) (1000 + Math.random() * 8999));
												Carriage vehicle = new Carriage(BASIC_TRUCKS.get(v));
												route.setCarriage(vehicle);

												/**
												 * 找到一个合理的分割线。
												 */
												// 记录所有的车内箱子的horizontal line
												ArrayList<Integer> inCar = new ArrayList<Integer>();// the boxes in
																									// car.
												ArrayList<Integer> outCar = new ArrayList<Integer>();// the boxes
																										// out car.
												double divide_line = line_before + BASIC_TRUCKS.get(v).getLength();// 刚开始设置的分割线。
												while (true) {
													// boolean check_support = false;//
													// 分割线没有将车外箱子分割时，不需要check_support.
													/**
													 * 根据分割线将箱子分割成车内和车外。
													 */
													inCar = new ArrayList<Integer>();// the boxes in car.
													outCar = new ArrayList<Integer>();// the boxes out car.
													double added_weight = 0.0;// 车内箱子的总重量。
													ArrayList<Double> horizontal_line = new ArrayList<Double>();
													for (int idx = 0; idx < goods.size(); idx++) {
														if (!is_added[idx]) {

															Box curr_box = goods.get(idx);
															// 最后端-最前端的长度<vehicle.length
															if (curr_box.getZCoor() >= line_before
																	&& curr_box.getZCoor()
																			+ curr_box.getLength() <= divide_line
																	&& Math.max(line_last,
																			curr_box.getZCoor()
																					+ curr_box.getLength())
																			- line_before <= BASIC_TRUCKS.get(v)
																					.getLength()
																	&& curr_box.getWeight()
																			+ added_weight <= BASIC_TRUCKS.get(v)
																					.getCapacity()) {
																inCar.add(idx);
																if (curr_box.getZCoor()
																		+ curr_box.getLength() > line_last)
																	line_last = curr_box.getZCoor()
																			+ curr_box.getLength();// 更新车内箱子的最后端。
																added_weight = added_weight + curr_box.getWeight();
																/**
																 * 将z+length加入到horizontal_line中，并且按照降序排列。
																 */
																horizontal_line.add(
																		curr_box.getZCoor() + curr_box.getLength());
															} else {
																outCar.add(idx);
																// if (curr_box.getZCoor() < divide_line
																// && curr_box.getZCoor() + curr_box
																// .getLength() > divide_line) {
																// check_support = true;
																// }
															}
														}
													}
													// Comparator.naturalOrder()升序。
													// Comparator.reverseOrder()降序。
													horizontal_line.sort(Comparator.reverseOrder());
													// for(int idx = horizontal_line.size()-1;idx>0;idx--) {
													/**
													 * 判断内外boxes相互不支撑。
													 */
													boolean support = false;
													// if(check_support) {
													/**
													 * 判断里面的箱子是否被外面的箱子支配。
													 */
													for (int in : inCar) {
														/**
														 * //计算该箱子的底部面积。 计算currBox被支撑的面积。
														 */
														Box currBox = goods.get(in);
														// double bottomArea = currBox.getWidth()
														// * currBox.getLength();
														double curr_y = currBox.getYCoor();
														double crossArea = 0;
														support = false;
														for (int out : outCar) {
															/**
															 * 判断车外箱子是否支撑车内箱子。
															 */
															// Box b_in = goods.get(in);
															// Box b_out = goods.get(out);
															Box existBox = goods.get(out);
															if (Math.abs(existBox.getYCoor() + existBox.getHeight()
																	- curr_y) <= 0.0) {
																double xc = currBox.getXCoor(),
																		zc = currBox.getZCoor(),
																		xe = existBox.getXCoor(),
																		ze = existBox.getZCoor();
																double wc = currBox.getWidth(),
																		lc = currBox.getLength(),
																		we = existBox.getWidth(),
																		le = existBox.getLength();

																if (!((xc + wc < xe) || (xe + we < xc)
																		|| (zc + lc < ze) || (ze + le < zc))) {
																	double[] XCoor = { xc, xc + wc, xe, xe + we };
																	double[] ZCoor = { zc, zc + lc, ze, ze + le };
																	// sort xc,xc+wc,xe,xe+we
																	Arrays.sort(XCoor);
																	Arrays.sort(ZCoor);
																	// sort zc,zc+lc,ze,ze+le
																	crossArea = crossArea
																			+ Math.abs(XCoor[2] - XCoor[1])
																					* Math.abs(ZCoor[2] - ZCoor[1]);
																	if (crossArea > 0) {
																		/**
																		 * 如果有支撑面积。
																		 */
																		support = true;
																		break;
																	}
																}
															}
														}
														if (support)
															break;
													}
													if (!support) {
														/**
														 * 判断外面的箱子是否被里面的箱子支配。
														 */
														for (int in : outCar) {
															/**
															 * //计算该箱子的底部面积。 计算currBox被支撑的面积。
															 */
															Box currBox = goods.get(in);
//																double bottomArea = currBox.getWidth()
//																		* currBox.getLength();
															double curr_y = currBox.getYCoor();
															double crossArea = 0;
															support = false;
															for (int out : inCar) {
																/**
																 * 判断车外箱子是否支撑车内箱子。
																 */
																// Box b_in = goods.get(in);
																// Box b_out = goods.get(out);
																Box existBox = goods.get(out);
																if (Math.abs(existBox.getYCoor()
																		+ existBox.getHeight() - curr_y) <= 0.0) {
																	double xc = currBox.getXCoor(),
																			zc = currBox.getZCoor(),
																			xe = existBox.getXCoor(),
																			ze = existBox.getZCoor();
																	double wc = currBox.getWidth(),
																			lc = currBox.getLength(),
																			we = existBox.getWidth(),
																			le = existBox.getLength();

																	if (!((xc + wc < xe) || (xe + we < xc)
																			|| (zc + lc < ze) || (ze + le < zc))) {
																		double[] XCoor = { xc, xc + wc, xe,
																				xe + we };
																		double[] ZCoor = { zc, zc + lc, ze,
																				ze + le };
																		// sort xc,xc+wc,xe,xe+we
																		Arrays.sort(XCoor);
																		Arrays.sort(ZCoor);
																		// sort zc,zc+lc,ze,ze+le
																		crossArea = crossArea + Math
																				.abs(XCoor[2] - XCoor[1])
																				* Math.abs(ZCoor[2] - ZCoor[1]);
																		if (crossArea > 0.0) {
																			/**
																			 * 如果有支撑面积。
																			 */
																			support = true;
																			break;
																		}
																	}
																}
															}
															if (support)
																break;
														}
													}
													// }
													if (!support&&inCar.size()>0) {
														/**
														 * 找到了可行的分割线。
														 */
														// divide_line
														break;
													} else {
														// 这条分割线不合理。那么找下一个分割线。
														int idx = 0;
														while (idx < horizontal_line.size()
																&& horizontal_line.get(idx) >= divide_line)
															idx++;// horizontal_line是从大到小排列的。
														if (idx < horizontal_line.size())
															divide_line = horizontal_line.get(idx);
														else {
															if (isLog)
																System.out.println("不能分割。");
															is_succ = false;
															break;
														}
														// break;//
													}
													// }//for each horizontal line
												}
												/**
												 * 判断分割线是否合理。 1. 车内total box weight < capacity. 2.
												 * 里外相互不支撑！！如果支撑，则其总的支撑面-里外支撑面>80% 如果不满足约束，则将分割线往里面移动一下。
												 */

												if (!is_succ) {
													break;
												}
												/**
												 * 处理boxes 确保没加入的boxes的z坐标>当前最后的line
												 * 
												 */
												ArrayList<Box> boxes_node0 = new ArrayList<Box>();
												/**
												 * 如果前面能够记录从第几个boxes到第几个box是node i的就好了。
												 * packing_result.getNode.get(0)
												 */
												double weight_temp = 0.0;
												double minz_load_box = Double.MAX_VALUE; // 相当于
												double maxz_load_box = Double.MIN_VALUE;
												for (int load : inCar) {
													if (goods.get(load).getZCoor() < minz_load_box) {
														minz_load_box = goods.get(load).getZCoor();
													}
													if (goods.get(load).getZCoor()
															+ goods.get(load).getLength() > maxz_load_box)
														maxz_load_box = goods.get(load).getZCoor()
																+ goods.get(load).getLength();
												}
												for (int load : inCar) {
													Box load_box = new Box(goods.get(load));
													load_box.setZCoor(goods.get(load).getZCoor() - minz_load_box);
													boxes_node0.add(load_box);
													weight_temp = weight_temp + load_box.getWeight();
													add_num = add_num + 1;
													is_added[load] = true;
												}
												/**
												 * 计算未加载箱子的最前端。
												 */
												line_before = Double.MAX_VALUE;
												for (int unload : outCar) {
													if (goods.get(unload).getZCoor() < line_before)
														line_before = goods.get(unload).getZCoor();
												}
												route.setBoxes(boxes_node0);// 这里先不设置boxes，注意在输出解的时候，我们要转换到route的boxes里面。
												Node divide_node = new Node(
														packing_result.getRoutes().get(v).getNodes().get(i));
												divide_node.setGoods(boxes_node0);
												divide_node.setDemands(maxz_load_box - minz_load_box);//
												divide_node.setWeights(weight_temp);

												LinkedList<Node> nodes = new LinkedList<Node>();
												nodes.add(new Node(depot_start));
												nodes.add(divide_node);
												nodes.add(new Node(depot_end));
												route.setNodes(nodes);
												route.setExcessLength(divide_node.getDemands());// 设置已经装载的总长度。
												route.setExcessWeight(divide_node.getWeights());
												if(PACKING_RECORD)
													route.getCarriage().setTruckTypeCode("pfrs"+pack_method);
												solution.getRoutes().add(route);
												/**
												 * 太多路径的就没必要了。车子太小了。
												 */
												// if(solution.getRoutes().size()>MAX_VEHICLES)
												// {is_succ=false;break;}
												/**
												 * 最后更新curr_line 更新
												 */
												// curr_line = curr_line+line_temp;
												// curr_box_idx = box_idx+1;//下一个要加载的箱子。
											}
											if (is_succ) {
												solutionSet_offspring.add(solution);
												n_times++;
											}

										} // for each feasible car.
											// }//new car选择装载率最高的车子。
									}
								} // for solutionSet，遍历所有的现有解。
								solutionSet.clear();
								for (Solution_vrp s : solutionSet_offspring.solutionList_)
									solutionSet.add(s);
								solutionSet_offspring.clear();
								if (isOutput)
									output(solutionSet, filenames[fileidx], PlatformIDCodeMap);
								if (isLog)
									System.out.println("finished load client " + i + " platform:"
											+ PlatformIDCodeMap.get(clients.get(i).getPlatformID()));

								if (solutionSet.size() > 1000) {
									// 进行减枝
									if (isLog)
										System.out.println("should be cut.");
								}
							} // for client.get(i)，遍历所有的client节点。
								// ********************************************************************************
							/**
							 * solutionSet的每个路径是否又更小的车子可以装载。 are there smaller vehicle to load each route.
							 */
							/**
							 * solutionSet经过一次brach and bound之后的结果。 将solutionSet转换成可以输出的格式，进行evaluation
							 */
							for (Solution_vrp s : solutionSet.solutionList_) {
								for (Route r : s.getRoutes()) {
									double curr_z = 0.0;
									ArrayList<Box> boxes = new ArrayList<Box>();
									for (Node n : r.getNodes()) {
										// 将node n的boxes加入到route.boxes里面。
										for (Box b : n.getGoods()) {
											Box new_b = new Box(b);
											new_b.setZCoor(b.getZCoor() + curr_z);
											boxes.add(new_b);
										}
										curr_z = curr_z + n.getDemands();
									}
									r.setBoxes(boxes);
								}
							}
							for (Solution_vrp s : solutionSet.solutionList_) {// 評價
								s.evaluation();
							}
							SolutionSet_vrp children = moveNode(solutionSet);
							for (Solution_vrp s : children.solutionList_) {
								s.evaluation();
								solutionSet.add(s);
							}
							/**
							 * 将solutionSet里面的加入到final_ss中去。
							 */
							solutionSet.removeDomintes();

							for (Solution_vrp s : solutionSet.solutionList_) {
								final_ss.add(new Solution_vrp(s));
							}
							// final_ss.removeDomintes();
							solutionSet.clear();

							if ((System.nanoTime() - begintime) / (1e9) > PFRS_TIME_LIMITS)
								break;
						} // for greedy
						if ((System.nanoTime() - begintime) / (1e9) > PFRS_TIME_LIMITS)
							break;
					} // packing_method 0,1,2,3,4
				} // box_sort_method 0,1
			 }
				System.out.printf("time(PFRS):%4.1f", (System.nanoTime() - begintime) / (1e9));
				
				
				
				
				
				if(true) {
					// main settings *************************************************
					boolean is_found_feasible = false;
				    int n_split = 4;
				    double[] relax_ratio_all = {0.95,0.925,0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.725,0.7,0.675,0.65,0.625,0.6,0.575,0.55,0.525,0.5,0.475,0.45,0.425, 0.4, 0.375, 0.35};
				    // main settings finished *************************************************

				for(int runs = 0; runs < n_repeat; runs++) {
//					System.out.println("run:"+runs);
			    	double [] if_hard_node_use = if_hard_node0;
			    	if (runs%3==0) {
//			    		System.out.print("run:"+runs+"hard.");
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
					    	split_minv = 0.5*VEHICLE_VOLUME[TRUCKTYPE_NUM-1];
					    }
					    else if (split_w==4){
					    	split_minv = 0.8*VEHICLE_VOLUME[TRUCKTYPE_NUM-1];
					    }
				    	//double split_minv =   ((double) split_w/ (double) n_split)*(max_v-min_v)+min_v-100.0; 
				    	
				    	problem = new CVRP_mix_integer("PermutationBinary",split_minv,VEHICLE_CAPACITY,VEHICLE_VOLUME,CLIENT_NUM,distanceMap,client_v_w,relax_ratio,truck_weight_ratio,clients,if_large_box,if_hard_node_use);
				
				    	algorithm = new cMOEAD(problem);
				
				        // Algorithm parameters
				        algorithm.setInputParameter("populationSize",POPSIZE);
				        algorithm.setInputParameter("maxEvaluations",maxE);
				        
				
//				        algorithm.setInputParameter("dataDirectory",
//				        "/Users/antelverde/Softw/pruebas/data/MOEAD_parameters/Weight");
				
//				        algorithm.setInputParameter("finalSize", 60) ; // used by MOEAD_DRA
				
				        algorithm.setInputParameter("T", T_) ;
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
				        
//				        algorithm.addOperator("crossover_end",crossover_end);
//				        algorithm.addOperator("mutation_end",mutation_end);
				        algorithm.addOperator("mutation_mod",mutation_mod);
				        
				        // Execute the Algorithm
//				        long initTime = System.currentTimeMillis();
				        SolutionSet population = algorithm.execute();
//				        population_last = population;
//				        long estimatedTime = System.currentTimeMillis() - initTime;

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
				        		if (save_objs.get(n)[0]>best_results.get(nn)[0]&&save_objs.get(n)[1]*relax_ratio*0.98>best_results.get(nn)[1]) {
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
//					        double[][] PF = (double[][]) PF_out.get("PF");
//					        int[] no_PF =(int[])PF_out.get("no_PF");

						    int[] save_feasible_no = new int[n_solution];
//						    System.out.println(""+runs+":"+n_solution);
						    for (int n = 0; n<n_solution; n++) {
						    		save_vars1.add(save_vars0.get(((int[])PF_out.get("no_PF"))[n]));
						    		save_objs1.add(save_objs0.get(((int[])PF_out.get("no_PF"))[n]));
						    }
						    
						    SolutionSet_vrp solutionSet = get_Solutions(split_minv,save_vars1,relax_ratio,truck_weight_ratio,save_feasible_no,if_hard_node_use);
						    
						    for (int n_solu=0;n_solu<solutionSet.size();n_solu++) {
//							    solutionSet_last.add(solutionSet.get(n_solu));
						    	allocateBoxes2Node(solutionSet.get(n_solu));
						    	final_ss.add(solutionSet.get(n_solu));
						    	is_found_feasible = true;
						    }
						    for (int n = 0; n<n_solution; n++) {
						    	if(save_feasible_no[n]==1) {
						    		double[] save_mid = save_objs0.get(n);
						    		save_mid[1] = save_mid[1]*relax_ratio;
						    		best_results.add(save_mid);
//						    		best_results.add(save_objs0.get(n));				    		
						    	}
						    }
						    
				        }//if save_objs>0
				        if(n_3DPacking>=PACKING_LIMITS || (n_3DPacking >= PACKING_LIMITS/2 && !is_found_feasible)||(System.nanoTime() - begintime) / (1e9) > RFPS_TIME_LIMITS) {
				        	break;
				        }
				    	}//for split_w
				        if(n_3DPacking>=PACKING_LIMITS || (n_3DPacking >= PACKING_LIMITS/2 && !is_found_feasible)||(System.nanoTime() - begintime) / (1e9) > RFPS_TIME_LIMITS) {
				        	break;
				        }
//				        if((System.nanoTime() - begintime) / (1e9) > RFPS_TIME_LIMITS) {
//				        	break;
//				        }
				    }//for relax_r
				    //System.out.println(runs);
			        if(n_3DPacking>=PACKING_LIMITS || (n_3DPacking >= PACKING_LIMITS/2 && !is_found_feasible)||(System.nanoTime() - begintime) / (1e9) > RFPS_TIME_LIMITS) {
			        	break;
			        }
//			        if((System.nanoTime() - begintime) / (1e9) > RFPS_TIME_LIMITS) {
//			        	break;
//			        }
//			        System.out.println(""+runs+":");
			    }
				System.out.printf("\t time:npack(RFPS):%.1f:%d",(System.nanoTime() - begintime) / (1e9),n_3DPacking);
			 }
				
				
				
				final_ss.removeDomintes();

//				long endtime = System.nanoTime();
//				double usedTime = (endtime - begintime) / (1e9);
//				System.out
//				.print("after track1 time:" + usedTime + "\t");
				if(is_calculate_hv)
				System.out.printf("\t hv(PFRS+RFPS):%4.3f", final_ss.get2DHV(idealNadir));
//				System.out.print("time(PFRS+RFPS):" + usedTime + "\t");
				if (debug)
					outputJSON(final_ss, filenames[fileidx], PlatformIDCodeMap,
							output_directory + '/' + filenames[fileidx]);
//			} // if track == 1

			/**
			 * end of track 1. and begin of track 2.
			 */
			// 结合相同两条路径相同平台的车子到同一个路径里面。
			final_ss.removeDomintes();
			if(is_calculate_hv) {
			double hv = final_ss.get2DHV(idealNadir);
			System.out.printf("\t hv:%4.3f", hv);
			}
			/**
			 * 从小车换大车来试试。
			 * for each route in solution, use vehicle with lower volume to replace it.
			 */
			boolean changed = false;
			for (Solution_vrp solution : final_ss.solutionList_) {
				changed = false;
				n_3DPacking++;// check for this solution
				for (int ri = 0; ri < solution.getRoutes().size(); ri++) {
					Route r = solution.getRoutes().get(ri);
					for (Carriage c : BASIC_TRUCKS) {
						if ((c.getTruckVolume() > r.getLoadVolume() && c.getCapacity() > r.getLoadWeight1())
								&& (c.getTruckVolume() < r.getCarriage().getTruckVolume() || // 这个车子要小于现在用的车子。
										c.getCapacity() < r.getCarriage().getCapacity())) {
							if (!isLoadable(c, r.getBoxes())) { continue; }
							// check if the new vehicle c can replace this vehicle.
							Route new_r = new Route(r);
							new_r.setCarriage(new Carriage(c));
							ArrayList<Integer> bpp = new_r.is_loadable(new_r.getBoxes());
							if (bpp.size() == r.getBoxes().size()) {
								// replace r by new_r
								solution.getRoutes().set(ri, new_r);
								changed=true;
							}
						}
					}
				}
				if(changed)
					solution.evaluation();
			}
			final_ss.removeDomintes();
			
			if (debug) {
				double hv = final_ss.get2DHV(idealNadir);
				System.out.println("hv before TSP: " + hv + "\t");
			}
			/**
			 * 检查每条路径是否有更好的遍历。
			 */
			for (Solution_vrp s : final_ss.solutionList_) {
				double isOpt;
				changed=false;
				n_3DPacking++;
				for (int ri = 0; ri < s.getRoutes().size(); ri++) {
					Route r = new Route(s.getRoutes().get(ri));
					allocateBoxes2Node(r);
					isOpt = findOpt(r);
					if (isOpt > 0) {
						ArrayList<Box> boxingSequence = new ArrayList<Box>();
						// 按r中nodes的顺序得到boxingSequence.
						// 读取boxes
						for (Node n : r.getNodes()) {
							for (Box b : n.getGoods())
								boxingSequence.add(b);// 调用之前得allocate2Node
						}
						ArrayList<Integer> k = r.is_loadable(boxingSequence);
						if (k.size() == boxingSequence.size()) {
							// System.out.println("a better routine is found!!!!");
							s.getRoutes().set(ri, r);
							changed = true;
						}
					}
				}
				if(changed)
					s.evaluation();
			}
			final_ss.removeDomintes();
			if (debug) {
				double hv = final_ss.get2DHV(idealNadir);
				System.out.println("hv AFTER TSP: " + hv + "\t");
				outputJSON(final_ss, filenames[fileidx], PlatformIDCodeMap, output_directory + '/' + filenames[fileidx]);
			}
			/**
			 * 从大车往小车换boxes，1v1,mv1,1vm，可以是一个节点换一个节点，也可以是多个节点换一个节点
			 * 1.V(大车）>V(小车），大车选的boxes要比小车多。 2.保持车类型不变，不会增加到更大车。 第一步：先选择小车的一个节点（体积和重量）
			 * 第二步：从大车里面选择节点 1） 大于所选择的小车的体积，2）小于小车的容量。选节点组合的方法：遍历每个节点，如果加入该节点满足约束，则加入。
			 * 第三步：按最优距离的路径进行装箱。如果成功，则生成新的解。
			 */
			/**
			 * 更加flexibility 清空一辆小车和大车，重新组合。 选择一个组合加入小车，新的组合的体积要大于原来的小车的体积。或者新的组合的总距离比原来小。
			 * 连接两个车的路径。0表示小车，1表示大车，适应值是小车的体积越大越好（满足约束）。
			 */
			move_nodes(final_ss);
			final_ss.removeDomintes();

			move_nodes_same(final_ss);
			final_ss.removeDomintes();
//			
			move_boxes(final_ss);
			final_ss.removeDomintes();
			if(test)
			final_ss.printObjectivesToFile("./Main9_plus/"+filenames[fileidx] + "_OBJ");
			outputJSON(final_ss, filenames[fileidx], PlatformIDCodeMap, output_directory + '/' + filenames[fileidx]);
			// /**************统计当前问题的结果**************/
			long endtime = System.nanoTime();
			double usedTime = (endtime - begintime) / (1e9);
			//// System.out.println();
			//// System.out.println("Program took："+usedTime+"s");

			if (is_calculate_hv) {
				double hv = final_ss.get2DHV(idealNadir);
				System.out.printf("\t finalhv:%4.3f", hv);
				total_hv = total_hv + hv;
			}
			System.out.print("\t solution num:" + final_ss.size());
			System.out.printf("\t time:%.1f s", usedTime);
			System.out.print("\t 3dbpp: " + n_3DPacking);
			total_time = total_time + usedTime;
			if(usedTime>max_single_time)
				max_single_time = usedTime;
			if(Math.floorMod(fileidx+1, 50)==0)
				System.out.println(new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date()));
			System.out.println();

			final_ss.clear();
			/**
			 * end of run one order.
			 */
		} // filename
		/**
		 * end of all run
		 */
		
		System.out.println("Total HV:\t" + total_hv);
		System.out.println("Total Time:\t" + total_time);
		System.out.println("max single time:"+max_single_time);
//		System.out.println("Total 3dbpp call:\t" + total_3dbpp_call);
		System.out.println(new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new Date()));
		// 保存所有idealNadirMap
		if (isUpdateExtreme)
			writeExtreme(idealNadirMap);
	}

	// =============================================================

	/**
	 * 判断结合后的两条路线是否有重叠的platform，
	 * 
	 * @param r1
	 *            路线1
	 * @param r2
	 *            路线2
	 * @return 返回重叠的platform的id
	 */
	public static ArrayList<Integer> isOverlap(Route r1, Route r2) {
		ArrayList<Integer> overlapIdx = new ArrayList<Integer>();
		for (int i = 1; i < r1.getNodes().size() - 1; i++) {
			for (int j = 1; j < r2.getNodes().size() - 1; j++) {
				if (r1.getNodes().get(i).getPlatformID() == r2.getNodes().get(j).getPlatformID()) {// 在r2中有r1_i这个node
					overlapIdx.add(r1.getNodes().get(i).getPlatformID());
				}
			}
		}
		return overlapIdx;
	}

	/**
	 * 
	 * @param r1
	 * @param r2
	 * @return 返回相同节点的下标。
	 */
	public static ArrayList<L> RoutesOverlap(Route r1, Route r2) {
		ArrayList<L> overlapIdx = new ArrayList<L>();
		for (int i = 1; i < r1.getNodes().size() - 1; i++) {
			for (int j = 1; j < r2.getNodes().size() - 1; j++) {
				if (r1.getNodes().get(i).getPlatformID() == r2.getNodes().get(j).getPlatformID()
						&& !r1.getNodes().get(i).isMustFirst()) {// 在r2中有r1_i这个node
					// Route r1 node i is same with Route r2 node j
					L l = new L();
					l.setI(i);
					l.setJ(j);
					overlapIdx.add(l);
				}
			}
		}
		return overlapIdx;
	}

	/**
	 * 将更新的ideal和nadir points保存到.\\data\\extremes
	 * 
	 * @param ideaNadirMap
	 */
	@SuppressWarnings("unchecked")
	public static void writeExtreme(Map<String, Double[]> ideaNadirMap) {

		// 创建json对象
		JSONObject extremesJSONObject = new JSONObject();
		JSONArray problemsArray = new JSONArray();
		File f = new File("./data/inputs");
		String[] filenames = f.list();
		for (int fileidx = 0; fileidx < filenames.length; fileidx++) {
			JSONObject problemObj = new JSONObject();
			problemObj.put("problemID", fileidx);
			problemObj.put("problemName", filenames[fileidx]);
			problemObj.put("outputFileForIdealF1", "./data/idealF1/" + filenames[fileidx]);
			problemObj.put("outputFileForIdealF2", "./data/idealF2/" + filenames[fileidx]);
			problemObj.put("outputFileForNadirF1", "./data/nadirF1/" + filenames[fileidx]);
			problemObj.put("outputFileForNadirF2", "./data/nadirF2/" + filenames[fileidx]);
			problemObj.put("idealF1", ideaNadirMap.get(filenames[fileidx])[0]);
			problemObj.put("idealF2", ideaNadirMap.get(filenames[fileidx])[1]);
			problemObj.put("nadirF1", ideaNadirMap.get(filenames[fileidx])[2]);
			problemObj.put("nadirF2", ideaNadirMap.get(filenames[fileidx])[3]);
			problemsArray.add(problemObj);
		}
		extremesJSONObject.put("problems", problemsArray);
		// 将json对象写入文件。
		try (FileWriter file = new FileWriter("./data/extremes")) {
			file.write(extremesJSONObject.toJSONString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 从.\\data\\extremes里面读取ideal and nadir points.
	 * 
	 * @return
	 */
	public static Map<String, Double[]> readExtreme() {
		Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();// problemName->ideal&nadir
		// JSON parser object to parse read file
		JSONParser jsonParser = new JSONParser();
		try (FileReader reader = new FileReader("./data/extremes")) {
			JSONObject obj = (JSONObject) jsonParser.parse(reader);// 最顶层

			JSONArray problemsArray = (JSONArray) obj.get("problems");
			Iterator<JSONObject> iterator = problemsArray.iterator();// 用来遍历JSONArray中的JSONObject
			while (iterator.hasNext()) {
				JSONObject curr_problem = iterator.next();
				Double[] idealNadirValues = { (Double) curr_problem.get("idealF1"),
						(Double) curr_problem.get("idealF2"), (Double) curr_problem.get("nadirF1"),
						(Double) curr_problem.get("nadirF2") };
				idealNadirMap.put((String) curr_problem.get("problemName"), idealNadirValues);
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

	/**
	 * 判断solution是否可行。
	 * 
	 * @param solution
	 * @param problem
	 * @param PlatformIDCodeMap
	 * @return
	 * @throws IOException
	 */
	public static boolean check_feasible(Solution_vrp solution, String problem, Map<Integer, String> PlatformIDCodeMap)
			throws IOException {
		outputJSON(solution, problem, ".\\data\\temp\\" + problem);
		Check orderCheck = Check.getOrderCheck(".\\data\\inputs", ".\\data\\temp", problem);
		if (!orderCheck.check()) {
			ArrayList<String> errorMessages = orderCheck.getErrorMessages();
			if (!errorMessages.isEmpty()) {
				System.out.println("the solution saved in temp is not feasible:" + problem);
				for (String errorMessage : errorMessages) {
					System.out.println(errorMessage);
				}
				System.out.println("");
			}
			return false;
		}
		return true;
	}

	/**
	 * 更新ideal和nadir points
	 * 
	 * @param solution
	 * @param idealNadir
	 * @param filenames
	 * @param fileidx
	 * @param PlatformIDCodeMap
	 * @return
	 * @throws IOException
	 */
	// 注意：传入的必须是可行解。
	public static boolean update_idealNadir(Solution_vrp solution, Double[] idealNadir, String[] filenames, int fileidx,
			Map<Integer, String> PlatformIDCodeMap) throws IOException {
		outputJSON(solution, filenames[fileidx], ".\\data\\temp\\" + filenames[fileidx]);
		Check orderCheck = Check.getOrderCheck(".\\data\\inputs", ".\\data\\temp", filenames[fileidx]);
		if (!orderCheck.check()) {
			ArrayList<String> errorMessages = orderCheck.getErrorMessages();
			if (!errorMessages.isEmpty()) {
				System.out.println("the solution saved in temp is not feasible:" + filenames[fileidx]);
				// for (String errorMessage: errorMessages) {
				// System.out.println(errorMessage);
				// }
				System.out.println("");
			}
			return false;
		}
		if (solution.getF1() < idealNadir[0]) {
			idealNadir[0] = solution.getF1();// minF1
			outputJSON(solution, filenames[fileidx], ".\\data\\idealF1\\" + filenames[fileidx]);
			System.out.print("new idealF1 is found.");
		}
		if (solution.getF2() < idealNadir[1]) {
			idealNadir[1] = solution.getF2();// minF2
			outputJSON(solution, filenames[fileidx], ".\\data\\idealF2\\" + filenames[fileidx]);
			System.out.print("new idealF2 is found.");
		}
		if (solution.getF1() > idealNadir[2]) {
			idealNadir[2] = solution.getF1();// maxF1
			outputJSON(solution, filenames[fileidx], ".\\data\\nadirF1\\" + filenames[fileidx]);
			System.out.print("new nadirF1 is found.");
		}
		if (solution.getF2() > idealNadir[3]) {
			idealNadir[3] = solution.getF2();// maxF2
			outputJSON(solution, filenames[fileidx], ".\\data\\nadirF2\\" + filenames[fileidx]);
			System.out.print("new nadirF2 is found.");
		}
		return true;
	}

	/**
	 * 将route.boxes分配到每个节点上。使其一一对应。
	 * 
	 * @param incumbentSolution
	 */
	public static void allocateBoxes2Node(Solution_vrp incumbentSolution) {
		for (int routei = 0; routei < incumbentSolution.getRoutes().size(); routei++) {// 对每条路径。
			Route currRoute = new Route(incumbentSolution.getRoutes().get(routei));
			Iterator<Box> allBoxesIterator = currRoute.getBoxes().iterator();// 遍历这条路径上的所有节点。
			int nodei = 1;
			ArrayList<Box> goods = new ArrayList<Box>();
			while (allBoxesIterator.hasNext()) {
				Box currBox = allBoxesIterator.next();
				if (currBox.getPlatformid() == currRoute.getNodes().get(nodei).getPlatformID()) {
					goods.add(new Box(currBox));
				} else {
					incumbentSolution.getRoutes().get(routei).getNodes().get(nodei).setGoods(goods);// 当前解不是这个平台的了。
					nodei = nodei + 1;// 下一个节点平台。
					goods = new ArrayList<Box>();// 清空goods
					goods.add(new Box(currBox));
				}
			}
			incumbentSolution.getRoutes().get(routei).getNodes().get(nodei).setGoods(goods);// 当前解不是这个平台的了。
		}
	}

	/**
	 * 
	 * @param currRoute
	 */
	public static void allocateBoxes2Node(Route currRoute) {
		Iterator<Box> allBoxesIterator = currRoute.getBoxes().iterator();// 遍历这条路径上的所有节点。
		int nodei = 1;
		ArrayList<Box> goods = new ArrayList<Box>();
		int goodsNum = 0;
		double goodsVolumn = 0.0;
		double goodsWeight = 0.0;

		while (allBoxesIterator.hasNext()) {
			Box currBox = allBoxesIterator.next();
			if (currBox.getPlatformid() == currRoute.getNodes().get(nodei).getPlatformID()) {
				goods.add(new Box(currBox));
				goodsNum += 1;
				goodsVolumn += currBox.getVolume();
				goodsWeight += currBox.getWeight();
			} else {
				currRoute.getNodes().get(nodei).setGoodsNum(goodsNum);
				currRoute.getNodes().get(nodei).setGoodsVolumn(goodsVolumn);
				currRoute.getNodes().get(nodei).setGoodsWeight(goodsWeight);
				currRoute.getNodes().get(nodei).setGoods(goods);// 当前解不是这个平台的了。
				nodei = nodei + 1;// 下一个节点平台。
				goods = new ArrayList<Box>();// 清空goods
				goods.add(new Box(currBox));
				goodsNum = 1;
				goodsVolumn = currBox.getVolume();
				goodsWeight = currBox.getWeight();
			}
		}
		currRoute.getNodes().get(nodei).setGoodsNum(goodsNum);
		currRoute.getNodes().get(nodei).setGoodsVolumn(goodsVolumn);
		currRoute.getNodes().get(nodei).setGoodsWeight(goodsWeight);
		currRoute.getNodes().get(nodei).setGoods(goods);// 当前解不是这个平台的了。
		if (nodei + 2 != currRoute.getNodes().size())
			System.out.println("error in allocateBox2Node!!!");
	}

	/**
	 * 初步判断（长宽高）箱子是否能够装到车子里面。
	 * 
	 * @param c，使用的车子。
	 * @param bl,
	 *            box list: 需要装载的箱子。
	 * @return
	 */
	public static boolean isLoadable(Carriage c, ArrayList<Box> bl) {
		// 判断所有未装载的箱子是否能够装进当前车子里。
		boolean flag = true;
		for (int boxi = 0; boxi < bl.size(); boxi++) {
			if (bl.get(boxi).getHeight() > c.getHeight() || bl.get(boxi).getLength() > c.getLength()
					|| bl.get(boxi).getWidth() > c.getWidth()) {
				flag = false;
				break;
			}
		}
		return flag;
	}

	/**
	 * 1.把solutionSet里面的nodes上的所有boxes加入到route.boxes里面，然后再output. 在track1中使用。
	 * 
	 * @param solutionSet
	 * @param filename
	 * @param PlatformIDCodeMap
	 */
	public static void output(SolutionSet_vrp solutionSet, String filename, Map<Integer, String> PlatformIDCodeMap) {
		for (Solution_vrp s : solutionSet.solutionList_) {
			for (Route r : s.getRoutes()) {
				double curr_z = 0.0;
				ArrayList<Box> boxes = new ArrayList<Box>();
				for (Node n : r.getNodes()) {
					/**
					 * 将node n的boxes加入到route.boxes里面。
					 * 
					 */
					for (Box b : n.getGoods()) {
						Box new_b = new Box(b);
						new_b.setZCoor(b.getZCoor() + curr_z);
						boxes.add(new_b);
					}
					curr_z = curr_z + n.getDemands();
				}
				r.setBoxes(boxes);
			}
		}
		outputJSON(solutionSet, filename, PlatformIDCodeMap, ".\\data\\outputs\\" + filename);
	}

	/**
	 * 将一个解输出到json文件中。
	 * 
	 * @param solution
	 * @param instanceName
	 * @param PlatformIDCodeMap
	 * @param outputFile
	 */
	@SuppressWarnings("unchecked")
	static void outputJSON(Solution_vrp solution, String instanceName,String outputFile) {
		// output to json file.
		JSONObject outputJSONObject = new JSONObject();
		// {
		// "estimateCode":"E1594518281316",
		outputJSONObject.put("estimateCode", instanceName);
		// ***************************************************准备truckArray
		JSONArray truckArray = new JSONArray();

		// Iterator<Route> iteratorRoute = solution.getRoutes().iterator();
		// while(iteratorRoute.hasNext()) {
		for (int routei = 0; routei < solution.getRoutes().size(); routei++) {
			// Carriage currTruck = .getCarriage();
			Route route = solution.getRoutes().get(routei);
			// 一辆车
			JSONObject truckJSONObject = new JSONObject();
			// 这辆车基本信息 1. innerHeight
			truckJSONObject.put("innerHeight", route.getCarriage().getHeight());

			// 这辆车经过的路径信息->2. platformArray
			ArrayList<String> platformArray = new ArrayList<String>();
			for (int nodei = 1; nodei < route.getNodes().size() - 1; nodei++) {
				platformArray.add(PlatformIDCodeMap.get(route.getNodes().get(nodei).getPlatformID()));
			}
			//
			// while(iterator.hasNext())
			// if(iterator.)
			//
			truckJSONObject.put("platformArray", platformArray);
			truckJSONObject.put("volume", route.getLoadVolume());
			truckJSONObject.put("innerWidth", route.getCarriage().getWidth());
			truckJSONObject.put("truckTypeCode", route.getCarriage().getTruckTypeCode());
			truckJSONObject.put("piece", route.getBoxes().size());// number of boxes
			JSONArray spuArray = new JSONArray();// the boxes array

			// Iterator<Box> iteratorBox = route.getCarriage().getBoxes().iterator();
			int order = 1;
			// while(iteratorBox.hasNext()) {
			for (int boxi = 0; boxi < route.getBoxes().size(); boxi++) {
				Box box = route.getBoxes().get(boxi);// iteratorBox.next();
				JSONObject currBox = new JSONObject();// the current box information
				currBox.put("spuId", box.getSpuBoxID());
				currBox.put("order", order);
				order = order + 1;
				currBox.put("direction", box.getDirection());// length parallel to the vehicle's length
				currBox.put("x", box.getXCoor() + box.getWidth() / 2.0 - route.getCarriage().getWidth() / 2.0);// -box.getWidth()
				currBox.put("y", box.getYCoor() + box.getHeight() / 2 - route.getCarriage().getHeight() / 2.0);//
				currBox.put("length", box.getLength());
				currBox.put("weight", box.getWeight());
				currBox.put("height", box.getHeight());
				currBox.put("width", box.getWidth());
				currBox.put("platformCode", PlatformIDCodeMap.get(box.getPlatformid()));
				currBox.put("z", box.getZCoor() + box.getLength() / 2 - route.getCarriage().getLength() / 2.0);// -box.getHeight()
				spuArray.add(currBox);
			}
			truckJSONObject.put("spuArray", spuArray);// the array of boxes
			truckJSONObject.put("truckTypeId", route.getCarriage().getTruckTypeId());
			truckJSONObject.put("innerLength", route.getCarriage().getLength());
			truckJSONObject.put("maxLoad", route.getCarriage().getCapacity());
			truckJSONObject.put("weight", route.getLoadWeight());
			truckArray.add(truckJSONObject);
		}
		// ***************************************************
		outputJSONObject.put("truckArray", truckArray);

		File f = new File(outputFile);
		if (!f.getParentFile().exists()) {
			f.getParentFile().mkdir();
		}
		// ".\\data\\outputs"+used_truck_id+"\\"+instanceName
		try (FileWriter file = new FileWriter(outputFile)) {
			file.write(outputJSONObject.toJSONString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 将一个解集输出到json文件中。
	 * 
	 * @param solutionSet
	 * @param instanceName
	 * @param PlatformIDCodeMap
	 * @param outputFile
	 */
	@SuppressWarnings("unchecked")
	static void outputJSON(SolutionSet_vrp solutionSet, String instanceName, Map<Integer, String> PlatformIDCodeMap,
			String outputFile) {
		// output to json file.
		JSONObject outputJSONObject = new JSONObject();
		// {
		// "estimateCode":"E1594518281316",
		outputJSONObject.put("estimateCode", instanceName);

		ArrayList<JSONArray> solutionArray = new ArrayList<JSONArray>();
		for (int solutioni = 0; solutioni < solutionSet.size(); solutioni++) {
			Solution_vrp solution = solutionSet.get(solutioni);
			// Iterator<Route> iteratorRoute = solution.getRoutes().iterator();
			// while(iteratorRoute.hasNext()) {
			// ***************************************************准备truckArray
			JSONArray truckArray = new JSONArray();
			for (int routei = 0; routei < solution.getRoutes().size(); routei++) {
				// Carriage currTruck = .getCarriage();
				Route route = solution.getRoutes().get(routei);
				// 一辆车
				JSONObject truckJSONObject = new JSONObject();
				// 这辆车基本信息 1. innerHeight
				truckJSONObject.put("innerHeight", route.getCarriage().getHeight());

				// 这辆车经过的路径信息->2. platformArray
				ArrayList<String> platformArray = new ArrayList<String>();
				for (int nodei = 1; nodei < route.getNodes().size() - 1; nodei++) {
					platformArray.add(PlatformIDCodeMap.get(route.getNodes().get(nodei).getPlatformID()));
				}
				//
				// while(iterator.hasNext())
				// if(iterator.)
				//
				truckJSONObject.put("platformArray", platformArray);
				truckJSONObject.put("volume", route.getLoadVolume());
				truckJSONObject.put("innerWidth", route.getCarriage().getWidth());
				truckJSONObject.put("truckTypeCode", route.getCarriage().getTruckTypeCode());
				truckJSONObject.put("piece", route.getBoxes().size());// number of boxes
				JSONArray spuArray = new JSONArray();// the boxes array

				// Iterator<Box> iteratorBox = route.getCarriage().getBoxes().iterator();
				int order = 1;
				// while(iteratorBox.hasNext()) {
				for (int boxi = 0; boxi < route.getBoxes().size(); boxi++) {
					Box box = route.getBoxes().get(boxi);// iteratorBox.next();
					JSONObject currBox = new JSONObject();// the current box information
					currBox.put("spuId", box.getSpuBoxID());
					currBox.put("order", order);
					order = order + 1;
					currBox.put("direction", box.getDirection());// length parallel to the vehicle's length
					currBox.put("x", box.getXCoor() + box.getWidth() / 2.0 - route.getCarriage().getWidth() / 2.0);// -box.getWidth()
					currBox.put("y", box.getYCoor() + box.getHeight() / 2 - route.getCarriage().getHeight() / 2.0);//
					currBox.put("length", box.getLength());
					currBox.put("weight", box.getWeight());
					currBox.put("height", box.getHeight());
					currBox.put("width", box.getWidth());
					currBox.put("platformCode", PlatformIDCodeMap.get(box.getPlatformid()));
					currBox.put("z", box.getZCoor() + box.getLength() / 2 - route.getCarriage().getLength() / 2.0);// -box.getHeight()
					spuArray.add(currBox);
				}
				truckJSONObject.put("spuArray", spuArray);// the array of boxes
				truckJSONObject.put("truckTypeId", route.getCarriage().getTruckTypeId());
				truckJSONObject.put("innerLength", route.getCarriage().getLength());
				truckJSONObject.put("maxLoad", route.getCarriage().getCapacity());
				truckJSONObject.put("weight", route.getLoadWeight());
				truckArray.add(truckJSONObject);
			} // routei
			solutionArray.add(truckArray);
		} // solutioni
			// ***************************************************
		outputJSONObject.put("solutionArray", solutionArray);

		File f = new File(outputFile);
		if (!f.getParentFile().exists()) {
			f.getParentFile().mkdir();
		}
		// ".\\data\\outputs"+used_truck_id+"\\"+instanceName
		try (FileWriter file = new FileWriter(outputFile)) {
			file.write(outputJSONObject.toJSONString());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * 将一个节点的已经排好的箱子分割成车内和车外两部分。
	 * 
	 * @param goods
	 * @param length_bound
	 * @param weight_bound
	 * @return
	 */
	static boolean[] divide_node(ArrayList<Box> goods, double[] length_bound, double[] weight_bound) {
		boolean[] in_out_car = new boolean[goods.size()];

		ArrayList<Integer> inCar = new ArrayList<Integer>();// the boxes in car.
		ArrayList<Integer> outCar = new ArrayList<Integer>();// the boxes out car.
		double divide_line = length_bound[0];// 刚开始的分割线。
		// ArrayList<Box> goods = nodei.getGoods();//当前节点所有的boxes，这些boxes已经有坐标了。
		boolean is_succ = true;
		while (true) {
			/* 根据分割线将箱子分割成车内和车外。 */
			inCar = new ArrayList<Integer>();
			outCar = new ArrayList<Integer>();
			double outCar_start = Double.MAX_VALUE;
			double outCar_end = 0.0;
			double in_weight = 0.0;// weights in car.
			double out_weight = 0.0;// weights out car.
			ArrayList<Double> horizontal_line = new ArrayList<Double>();
			for (int idx = 0; idx < goods.size(); idx++) {
				Box curr_box = goods.get(idx);
				if (curr_box.getZCoor() + curr_box.getLength() <= divide_line
						&& curr_box.getWeight() + in_weight <= weight_bound[0]) {
					inCar.add(idx);
					in_weight = in_weight + curr_box.getWeight();
					horizontal_line.add(curr_box.getZCoor() + curr_box.getLength());
					in_out_car[idx] = false;
				} else {
					outCar.add(idx);
					out_weight = out_weight + curr_box.getWeight();
					if (curr_box.getZCoor() + curr_box.getLength() > outCar_end) {
						outCar_end = curr_box.getZCoor() + curr_box.getLength();
					}
					if (curr_box.getZCoor() < outCar_start) {
						outCar_start = curr_box.getZCoor();
					}
					in_out_car[idx] = true;
				}
			}

			if (outCar.size() > 0 && outCar_end - outCar_start <= length_bound[1] && out_weight <= weight_bound[1]) {
				/*** 判断内外boxes相互不支撑。 ***/
				boolean support = false;
				for (int in : inCar) {
					// 计算currBox被支撑的面积。
					Box currBox = goods.get(in);
					double curr_y = currBox.getYCoor();
					double crossArea = 0;
					support = false;
					for (int out : outCar) {
						Box existBox = goods.get(out);
						if (Math.abs(existBox.getYCoor() + existBox.getHeight() - curr_y) <= 0.0) {
							double xc = currBox.getXCoor(), zc = currBox.getZCoor(), xe = existBox.getXCoor(),
									ze = existBox.getZCoor();
							double wc = currBox.getWidth(), lc = currBox.getLength(), we = existBox.getWidth(),
									le = existBox.getLength();

							if (!((xc + wc < xe) || (xe + we < xc) || (zc + lc < ze) || (ze + le < zc))) {
								double[] XCoor = { xc, xc + wc, xe, xe + we };
								double[] ZCoor = { zc, zc + lc, ze, ze + le };
								// sort xc,xc+wc,xe,xe+we
								Arrays.sort(XCoor);
								Arrays.sort(ZCoor);
								// sort zc,zc+lc,ze,ze+le
								crossArea = crossArea + Math.abs(XCoor[2] - XCoor[1]) * Math.abs(ZCoor[2] - ZCoor[1]);
								if (crossArea > 0) {
									/*** 如果有支撑面积，不论大小。这个分割不合理。。 */
									support = true;
									break;
								}
							}
						}
					}
					if (support)
						break;
				}
				if (!support) {
					for (int in : outCar) {
						// 计算currBox被支撑的面积。
						Box currBox = goods.get(in);
						double curr_y = currBox.getYCoor();
						double crossArea = 0;
						support = false;
						for (int out : inCar) {
							Box existBox = goods.get(out);
							if (Math.abs(existBox.getYCoor() + existBox.getHeight() - curr_y) <= 0.0) {
								double xc = currBox.getXCoor(), zc = currBox.getZCoor(), xe = existBox.getXCoor(),
										ze = existBox.getZCoor();
								double wc = currBox.getWidth(), lc = currBox.getLength(), we = existBox.getWidth(),
										le = existBox.getLength();

								if (!((xc + wc < xe) || (xe + we < xc) || (zc + lc < ze) || (ze + le < zc))) {
									double[] XCoor = { xc, xc + wc, xe, xe + we };
									double[] ZCoor = { zc, zc + lc, ze, ze + le };
									// sort xc,xc+wc,xe,xe+we
									Arrays.sort(XCoor);
									Arrays.sort(ZCoor);
									// sort zc,zc+lc,ze,ze+le
									crossArea = crossArea
											+ Math.abs(XCoor[2] - XCoor[1]) * Math.abs(ZCoor[2] - ZCoor[1]);
									if (crossArea > 0) {
										/*** 如果有支撑面积，不论大小。这个分割不合理。。 */
										support = true;
										break;
									}
								}
							}
						}
						if (support)
							break;
					}
				}
				if (!support&&inCar.size()>0) {
					/** 找到了可行的分割线。 **/
					break;// break while true
				} else {
					// 这条分割线不合理。那么找下一个分割线。
					// horizontal_line是inCar里面的所有水平线。
					horizontal_line.sort(Comparator.reverseOrder());// 按降序排列。
					int idx = 0;// 下一条divide_line要比现在的要小。
					while (idx < horizontal_line.size() && horizontal_line.get(idx) >= divide_line)
						idx++;// horizontal_line是从大到小排列的。
					if (idx < horizontal_line.size())
						divide_line = horizontal_line.get(idx);
					else {
						if (isLog)
							System.out.println("不能分割。");
						is_succ = false;
						break;// break while true
					}
				}
			} else {// 满足长宽高和weight约束后。
					// 这条分割线不合理。那么找下一个分割线。
					// horizontal_line是inCar里面的所有水平线。
				horizontal_line.sort(Comparator.reverseOrder());// 按降序排列。
				int idx = 0;// 下一条divide_line要比现在的要小。
				while (idx < horizontal_line.size() && horizontal_line.get(idx) >= divide_line)
					idx++;// horizontal_line是从大到小排列的。
				if (idx < horizontal_line.size())
					divide_line = horizontal_line.get(idx);
				else {
					if (isLog)
						System.out.println("不能分割。");
					is_succ = false;
					break;// break while true
				}
			}
		} // while true
		if (!is_succ) {
			return null;
		} else {
			return in_out_car;
		}
	}

	/**
	 * 得到0，1，2，。。。，size-1的一个permutation。
	 * 
	 * @param perm
	 * @param size
	 */
	static void randomPermutation(int[] perm, int size) {
		int[] index = new int[size];
		boolean[] flag = new boolean[size];

		for (int n = 0; n < size; n++) {
			index[n] = n;
			flag[n] = true;
		}

		int num = 0;
		while (num < size) {
			int start = PseudoRandom.randInt(0, size - 1);
			while (true) {
				if (flag[start]) {
					perm[num] = index[start];
					flag[start] = false;
					num++;
					break;
				}
				if (start == (size - 1)) {
					start = 0;
				} else {
					start++;
				}
			}
		} // while
	} // randomPermutation

	/**
	 * 在track1场景中使用，solutionSET已经是排好的箱子。
	 * 
	 * @param solutionSet
	 * @return
	 */
	static SolutionSet_vrp moveNode(SolutionSet_vrp solutionSet) {
		SolutionSet_vrp children = new SolutionSet_vrp();
		for (Solution_vrp s : solutionSet.solutionList_) {
			ArrayList<L> LSequence = new ArrayList<L>();
			for (int ri = 0; ri < s.getRoutes().size(); ri++) {
				Route routei = s.getRoutes().get(ri);
				for (int ni = 1; ni < routei.getNodes().size() - 1; ni++) {
					Node nodei = routei.getNodes().get(ni);
					for (int rj = 0; rj < s.getRoutes().size(); rj++) {
						if (ri == rj || nodei.isMustFirst())
							continue;
						Route routej = s.getRoutes().get(rj);
						// 还需要检查nodei中的箱子是否可以被routej的车子装载。
						boolean height_ok = (routei.getCarriage().getHeight() <= routej.getCarriage().getHeight());
						boolean width_ok = (routei.getCarriage().getWidth() <= routej.getCarriage().getWidth());
						if (height_ok && width_ok
								&& routej.getCarriage().getLength() - routej.getExcessLength() >= nodei.getDemands()
								&& routej.getCarriage().getCapacity() - routej.getExcessWeight() >= nodei
										.getWeights()) {
							// 插入到距离最短的地方。
							// 加入到当前这个现有的车子里面中，找一个距离最小的地方插入。
							double min_increase = Double.MAX_VALUE;
							int insert_position = -1;// 插入到这个节点的后面。
							// 遍历当前可行路径的所有节点
							for (int k = 0; k < routej.getNodes().size() - 1; k++) {
								if (routej.getNodes().get(k + 1).isMustFirst())
									continue;
								/**
								 * 原来是0-1-2，则可以插入0或者1的后面。 原来是0-1-2-3，则可以插入0，1，2，的后面。
								 * 若是插入0后面，则0-new-1-2-3，则距离增加0-new-1减去0-1 找到距离增加最少的位置。
								 */
								String twoPlatform = String.valueOf(routej.getNodes().get(k).getPlatformID() + "+"
										+ routej.getNodes().get(k + 1).getPlatformID());
								// System.out.println(twoPlatform);
								double dist_before = distanceMap.get(twoPlatform);
								twoPlatform = String.valueOf(
										routej.getNodes().get(k).getPlatformID() + "+" + nodei.getPlatformID());
								double dist_after = distanceMap.get(twoPlatform);
								twoPlatform = String.valueOf(
										nodei.getPlatformID() + "+" + routej.getNodes().get(k + 1).getPlatformID());
								dist_after = dist_after + distanceMap.get(twoPlatform);
								if (dist_after - dist_before < min_increase) {
									min_increase = dist_after - dist_before;
									insert_position = k;
								}
							} // 遍历当前可行路径的所有节点
								// 记录加入到j路径的位置insert_position，新增距离是多少。
							L l = new L();
							l.setI(ri);
							l.setJ(rj);
							l.setN(ni);// ri中ni这个节点。插入到rj的insert_position
							l.setM(insert_position);
							l.setSij(min_increase);
							LSequence.add(l);
						}
						// 取ri中的一个节点与rj中的一个节点进行互换。
						// 取ri中的一个节点移动到rj中去，然后检查目标函数值。
						//
					}
				}
			}
			Collections.sort(LSequence);
			for (int il = 0; il < LSequence.size(); il++) {
				/* 产生新的解 */
				Solution_vrp new_solution = new Solution_vrp(s);
				int ri = LSequence.get(il).getI();
				int rj = LSequence.get(il).getJ();
				int ni = LSequence.get(il).getN();// ri中ni这个节点。插入到rj的insert_position
				int insert_position = LSequence.get(il).getM();
				// ri中ni这个节点。插入到rj的insert_position
				Node nodei = new Node(s.getRoutes().get(ri).getNodes().get(ni));
				new_solution.getRoutes().get(rj).getNodes().add(insert_position + 1, nodei);
				double excessLength = new_solution.getRoutes().get(rj).getExcessLength() + nodei.getDemands();
				double excessWeight = new_solution.getRoutes().get(rj).getExcessWeight() + nodei.getWeights();
				new_solution.getRoutes().get(rj).setExcessLength(excessLength);
				new_solution.getRoutes().get(rj).setExcessWeight(excessWeight);
				if (new_solution.getRoutes().get(ri).getNodes().size() == 3)
					new_solution.getRoutes().remove(ri);
				else {
					new_solution.getRoutes().get(ri).getNodes().remove(ni);
					excessLength = new_solution.getRoutes().get(ri).getExcessLength() - nodei.getDemands();
					excessWeight = new_solution.getRoutes().get(ri).getExcessWeight() - nodei.getWeights();
					new_solution.getRoutes().get(ri).setExcessLength(excessLength);
					new_solution.getRoutes().get(ri).setExcessWeight(excessWeight);
				}

				children.add(new_solution);
			}
		} // for each solution
			// for(Solution s:children.solutionList_)
			// final_ss.add(s);

		for (Solution_vrp s : children.solutionList_) {
			for (Route r : s.getRoutes()) {
				double curr_z = 0.0;
				ArrayList<Box> boxes = new ArrayList<Box>();
				for (Node n : r.getNodes()) {
					/**
					 * 将node n的boxes加入到route.boxes里面。
					 * 
					 */
					for (Box b : n.getGoods()) {
						Box new_b = new Box(b);
						new_b.setZCoor(b.getZCoor() + curr_z);
						boxes.add(new_b);
					}
					curr_z = curr_z + n.getDemands();
				}
				r.setBoxes(boxes);
			}
		}
		return children;
	}

	/**
	 * 在routej中找到插入platformID的最好的位置。
	 * 
	 * @param routej
	 * @param platformID
	 * @param distanceMap
	 * @return
	 */
	static double[] find_insertPos(Route routej, int platformID) {
		// int insertPos = -1;

		// 插入到距离最短的地方。
		// 加入到当前这个现有的车子里面中，找一个距离最小的地方插入。
		double min_increase = Double.MAX_VALUE;
		int insert_position = -1;// 插入到这个节点的后面。
		double[] return_pos_min_increase = new double[2];
		// 遍历当前可行路径的所有节点
		for (int k = 0; k < routej.getNodes().size() - 1; k++) {
			if (routej.getNodes().get(k + 1).isMustFirst())
				continue;
			/**
			 * 原来是0-1-2，则可以插入0或者1的后面。 原来是0-1-2-3，则可以插入0，1，2，的后面。
			 * 若是插入0后面，则0-new-1-2-3，则距离增加0-new-1减去0-1 找到距离增加最少的位置。
			 */
			String twoPlatform = String.valueOf(
					routej.getNodes().get(k).getPlatformID() + "+" + routej.getNodes().get(k + 1).getPlatformID());
			// System.out.println(twoPlatform);
			double dist_before = 0;
			if (distanceMap.containsKey(twoPlatform))
				dist_before = distanceMap.get(twoPlatform);
			else
				System.out.println(twoPlatform);
			twoPlatform = String.valueOf(routej.getNodes().get(k).getPlatformID() + "+" + platformID);
			double dist_after = distanceMap.get(twoPlatform);
			twoPlatform = String.valueOf(platformID + "+" + routej.getNodes().get(k + 1).getPlatformID());
			dist_after = dist_after + distanceMap.get(twoPlatform);
			if (dist_after - dist_before < min_increase) {
				min_increase = dist_after - dist_before;
				insert_position = k;
			}
		} // 遍历当前可行路径的所有节点
		return_pos_min_increase[0] = insert_position;
		return_pos_min_increase[1] = min_increase;
		return return_pos_min_increase;

	}

	/**
	 * 确保一条路径是最优路径。
	 * 
	 * @return -1表示输入已经是最优了。>0表示最优的路径的distance
	 */
	static double findOpt(Route r) {
		double new_dist = -1;
		int n = r.getNodes().size() - 2;
		int[] RouteIdx = new int[n];// {3,4,2,1,5};
		LinkedList<Node> nodes = r.getNodes();
		LinkedList<Node> nodes_new = new LinkedList<Node>();
		int[] indexes = new int[n];// 所有platform节点。
		for (int i = 0; i < n; i++) {
			indexes[i] = 0;
			RouteIdx[i] = i + 1;// platform nodes 0,1,...,n,n+1
		}
		double min_dist = calculateDist(nodes);
		double curr_dist = 0;
		int[] min_Route = null;
		if (n <= MAX_VRP) {// 4个节点24种情况。
			int i = 0;
			while (i < n) {
				if (indexes[i] < i) {
					swap(RouteIdx, i % 2 == 0 ? 0 : indexes[i], i);
					// printArray(originRoute);
					/** 构建新的nodes **/
					nodes_new.clear();
					nodes_new.add(nodes.get(0));
					for (int ri = 0; ri < n; ri++)
						nodes_new.add(nodes.get(RouteIdx[ri]));
					nodes_new.add(nodes.get(n + 1));
					curr_dist = calculateDist(nodes_new);
					if (curr_dist < min_dist) {
						boolean violateMustfirst = false;
						for (int ri = 2; ri < nodes_new.size() - 1; ri++)// 如果后面的节点有mustfirst约束，则不可行。
							if (nodes_new.get(ri).isMustFirst()) {
								violateMustfirst = true;
								break;
							}
						if (!violateMustfirst) {
							min_dist = curr_dist;
							min_Route = RouteIdx.clone();
							new_dist = min_dist;
						}
					}
					indexes[i]++;
					i = 0;
				} else {
					indexes[i] = 0;
					i++;
				}
			}
			if (new_dist > 0) {// 找到了新的更好路径。
				nodes_new.clear();
				nodes_new.add(nodes.get(0));
				for (int ri = 0; ri < n; ri++)
					nodes_new.add(nodes.get(min_Route[ri]));
				nodes_new.add(nodes.get(n + 1));
				// 覆盖
				r.setNodes(nodes_new);
			}
		}
		// 5个节点120个种情况
		// 6个节点720个情况
		// 7个节点5040个情况。
		else {
			// 2-opt method.
			System.out.println("findOpt: There are too many platforms in this route:" + n);
			// System.exit(0);
		}
		return new_dist;
	}

	/**
	 * 只是计算这些节点的最优路径。
	 * 
	 * @param nodes，要包含起始和结束节点。
	 * @param distanceMap
	 * @return
	 */
	static double bestVRP(LinkedList<Node> nodes) {
		int n = nodes.size() - 2;
		if (n == 0)
			return 0.0;
		int[] RouteIdx = new int[n];// {3,4,2,1,5};
		LinkedList<Node> nodes_new = new LinkedList<Node>();
		int[] indexes = new int[n];// 所有platform节点。
		for (int i = 0; i < n; i++) {
			indexes[i] = 0;
			RouteIdx[i] = i + 1;// platform nodes 0,1,...,n,n+1
		}
		double min_dist = calculateDist(nodes);
		double curr_dist = 0;
		// int[] min_Route = null;
		if (n <= MAX_VRP) {// 4个节点24种情况。
			int i = 0;
			while (i < n) {
				if (indexes[i] < i) {
					swap(RouteIdx, i % 2 == 0 ? 0 : indexes[i], i);
					// printArray(originRoute);
					/** 构建新的nodes **/
					nodes_new.clear();
					nodes_new.add(nodes.get(0));
					for (int ri = 0; ri < n; ri++)
						nodes_new.add(nodes.get(RouteIdx[ri]));
					nodes_new.add(nodes.get(n + 1));
					curr_dist = calculateDist(nodes_new);
					if (curr_dist < min_dist) {
						boolean violateMustfirst = false;
						for (int ri = 2; ri < nodes_new.size() - 1; ri++)// 如果后面的节点有mustfirst约束，则不可行。
							if (nodes_new.get(ri).isMustFirst()) {
								violateMustfirst = true;
								break;
							}
						if (!violateMustfirst) {
							// if(debug)
							// System.out.println("found a new better rotue!!!!!");
							min_dist = curr_dist;
							// min_Route = RouteIdx.clone();
							// new_dist = min_dist;
						}
					}
					indexes[i]++;
					i = 0;
				} else {
					indexes[i] = 0;
					i++;
				}
			}
			// if(new_dist>0) {//找到了新的更好路径。
			// nodes_new.clear();
			// nodes_new.add(nodes.get(0));
			// for(int ri=0;ri<n;ri++) nodes_new.add(nodes.get(min_Route[ri]));
			// nodes_new.add(nodes.get(n+1));
			// //覆盖
			// r.setNodes(nodes_new);
			// }
			// return min_dist;
		}
		// 5个节点120个种情况
		// 6个节点720个情况
		// 7个节点5040个情况。
		else {
			// 2-opt method.
			System.out.println("bestVRP: There are two many platforms in this route:" + n);
			// System.exit(0);
		}
		return min_dist;
	}

	/**
	 * 返回N条最优路径。
	 * 
	 * @param r
	 * @param distanceMap
	 * @return
	 */
	static ArrayList<L> findOptN(Route r) {
		ArrayList<L> all_routes = new ArrayList<L>();

		boolean violatemustfirst;
		int n = r.getNodes().size() - 2;
		int[] RouteIdx = new int[n];// {3,4,2,1,5};
		LinkedList<Node> nodes = r.getNodes();
		LinkedList<Node> nodes_new = new LinkedList<Node>();
		if (n <= 0)
			return all_routes;
		int[] indexes = new int[n];// 所有platform节点。
		for (int i = 0; i < n; i++) {
			indexes[i] = 0;
			RouteIdx[i] = i + 1;// platform nodes 0,1,...,n,n+1
		}
		double min_dist = calculateDist(nodes);
		L l = new L();
		l.setOverlapIdx(RouteIdx);
		l.setSij(-min_dist);
		all_routes.add(l);
		double curr_dist = 0;
		int[] min_Route = null;
		if (n <= MAX_VRP) {// 4个节点24种情况。
			int i = 0;
			while (i < n) {
				if (indexes[i] < i) {
					swap(RouteIdx, i % 2 == 0 ? 0 : indexes[i], i);
					// printArray(originRoute);
					/** 构建新的nodes **/
					nodes_new.clear();
					nodes_new.add(nodes.get(0));
					for (int ri = 0; ri < n; ri++)
						nodes_new.add(nodes.get(RouteIdx[ri]));
					nodes_new.add(nodes.get(n + 1));
					curr_dist = calculateDist(nodes_new);
					violatemustfirst = false;
					for (int ri = 2; ri < nodes_new.size() - 1; ri++)// 如果后面的节点有mustfirst约束，则不可行。
						if (nodes_new.get(ri).isMustFirst()) {
							violatemustfirst = true;
							break;
						}
					if (!violatemustfirst) {
						l = new L();
						l.setOverlapIdx(RouteIdx);
						l.setSij(-curr_dist);
						all_routes.add(l);
					}
					indexes[i]++;
					i = 0;
				} else {
					indexes[i] = 0;
					i++;
				}
			}
			// if(new_dist>0) {//找到了新的更好路径。
			// nodes_new.clear();
			// nodes_new.add(nodes.get(0));
			// for(int ri=0;ri<n;ri++) nodes_new.add(nodes.get(min_Route[ri]));
			// nodes_new.add(nodes.get(n+1));
			// //覆盖
			// r.setNodes(nodes_new);
			// }
		}
		// 5个节点120个种情况
		// 6个节点720个情况
		// 7个节点5040个情况。
		else {
			// 2-opt method.
			System.out.println("findOptN:There are two many platforms in this route:" + n);
			// System.exit(0);
		}
		Collections.sort(all_routes);// 按sij从大到小排序。
		while (all_routes.size() > NUM_ROUTE_GREEDY_TEST) {
			all_routes.remove(all_routes.size() - 1);
		}
		return all_routes;
	}

	static void swap(int[] input, int a, int b) {
		int tmp = input[a];
		input[a] = input[b];
		input[b] = tmp;
	}

	static double calculateDist(LinkedList<Node> nodes) {
		double dist = 0.0;
		for (int i1 = 0; i1 < nodes.size() - 1; i1++) {//
			String twoPlatform = String.valueOf(nodes.get(i1).getPlatformID()) + '+'
					+ String.valueOf(nodes.get(i1 + 1).getPlatformID());
			// distance+=distanceMap.get(twoPlatform);
			// if(this.distanceMap.containsKey(twoPlatform))
			dist += distanceMap.get(twoPlatform);// caculateDistance(route.getNodes().get(i1).getPlatformID(),
													// route.getNodes().get(i1+1).getPlatformID());
			// else
			// System.out.println(twoPlatform);
			// System.exit(0);
		}
		return dist;
	}

	static void move_boxes(SolutionSet_vrp solutionSet) {
		/**
		 * 精细搜索（需要时间较长） 从大车移动箱子到小车。 对node进行拆分。在可移动的情况下，移动越多越好。
		 * 1.选择一个node进行拆分。1.如果有相同的node，则选择相同的node进行移动，2.如果没有，则选择距离增加最少的节点，
		 * 哪些箱子需要移动呢？列出一系列箱子组合，0-移动，1-留下，移动的越多越好（在约束范围内，也就是车子的载重和容量。）
		 * 这里注意，不会全移动，也不会全留下。至少移动一个。
		 */
		for (int si = 0; si < solutionSet.size(); si++) {
			Solution_vrp solution = new Solution_vrp(solutionSet.get(si));
			if (debug) {
				System.out.println("before move boxes: f1: " + solution.getF1() + ";f2:" + solution.getF2());
				// outputJSON(solutionSet, filenames[fileidx], PlatformIDCodeMap,
				// output_directory+'/'+filenames[fileidx]);
			}
			Route routei, routej;
			// double min_increase, curr_increase;
			int selected_node = -1;
			boolean is_succ = false;
			ArrayList<Box> movedBox = new ArrayList<Box>();
			ArrayList<Box> boxesj;
			Node nodej;
			int checkidx = -1;// 用来检查哪条路径可以装载。从最优到次优进行检查。
			for (int ri = 0; ri < solution.getRoutes().size(); ri++) {
				routei = solution.getRoutes().get(ri);// routei是小车。
				for (int rj = 0; rj < solution.getRoutes().size(); rj++) {// 找到大车
					routej = solution.getRoutes().get(rj);// routej是大车。
					if (ri == rj || routej.getCarriage().getCapacity() <= routei.getCarriage().getCapacity()) {
						continue;
					}
					// 1.选择大车里面的一个node，mustfirst node不进行移动。
					// 最简单的随机选择一个node
					// 选择距离增加最少的node，如果拆分这个node的boxes，距离增加最少。
					// 选择的node必须有boxes>=2
					// 为route1增加node，距离增加了多少？
					// min_increase = Double.MAX_VALUE;
					for (selected_node = 1; selected_node < routej.getNodes().size() - 1; selected_node++) {
						nodej = routej.getNodes().get(selected_node);
						boxesj = nodej.getGoods();
						// 2.选择好node之后，进行boxes的选择。
						int n = nodej.getGoods().size();// number of boxes
						int[] x = new int[n];
						int[] xbest = new int[n];
						double volumni_best = routei.getLoadVolumn();
						double weight_best = routei.getLoadWeight();
						double volumni = 0.0, volumnj = 0.0, weighti = 0.0, weightj = 0.0;
						double volumni_currBest = routei.getLoadVolume();//routei原来装了多少。
						for (int xi = 0; xi < n; xi++) {
							x[xi] = 1;
							xbest[xi] = 1;//刚开始的boxes都在routej的selected_node里面。
						}
						ArrayList<L> betterCombination = new ArrayList<L>();
						boolean changed = false;
						if (n <= MOVE_BOX_BRUTEFORCE) {// 全部遍历。
							int[][] allx = new int[(int) Math.pow(2, n)][n];
							// 去除全0和全1的解。，计算0对应箱子的体积。
							// 全0对应着将routej这个节点的箱子都移动到routei里面。全1表示不变。
							for (int i = 1; i < (int) Math.pow(2, n) - 1; i++) {
								String curr = Integer.toBinaryString(i);
								int char_idx = curr.length() - 1;
								volumni = 0.0;
								weighti = 0.0;
								volumnj = 0.0;
								weightj = 0.0;
								movedBox.clear();
								for (int idx = n - 1; idx >= 0; idx--) {
									if (char_idx >= 0) {
										allx[i][idx] = Character.getNumericValue(curr.charAt(char_idx));
										char_idx--;
									} else
										allx[i][idx] = 0;
									if (allx[i][idx] == 0) {// node idx is load by routei
										movedBox.add(nodej.getGoods().get(idx));
										volumni += boxesj.get(idx).getVolume();
										weighti += boxesj.get(idx).getWeight();
									}
								}
								if (isLoadable(routei.getCarriage(), movedBox)
										&& volumni_best + volumni < routei.getCarriage().getTruckVolume()
										&& weight_best + weighti < routei.getCarriage().getCapacity()) {// routej重量约束
									L l = new L();
									l.setOverlapIdx(allx[i]);
									l.setSij(volumni + volumni_best);
									betterCombination.add(l);
									changed = true;
								}
							}
						} else {// 用进化算法来求解一个组合。(bit-flip)
							int iter = 0;
							boolean has_zero = false, has_one=false;
							while (iter < 100) {
								Random rand = new Random(System.currentTimeMillis());
								volumni = 0.0;
								weighti = 0.0;
								movedBox.clear();//避免全0和全1的解。
								has_zero = false;
								for (int xi = 0; xi < n; xi++) {
									if (rand.nextDouble() < BIT_FLIP_MOVE_BOXES) {// disturb with probability 0.2
										if (x[xi] == 0) {
											x[xi] = 1;
											has_one = true;
										} else {
											x[xi] = 0;
											has_zero = true;
											weighti += boxesj.get(xi).getWeight();
											volumni += boxesj.get(xi).getVolume();
										}
									} else {// keep the best.
										x[xi] = xbest[xi];
										if (x[xi] == 0) {
											has_zero = true;
											weighti += boxesj.get(xi).getWeight();
											volumni += boxesj.get(xi).getVolume();
										} else {
											has_one = true;
											// volumnj+=nodes.get(xi).getGoodsVolumn();
											// weightj+=nodes.get(xi).getGoodsWeight();
										}
									}
								}
								// //select
								if (isLoadable(routei.getCarriage(), movedBox) && has_zero && has_one
										&& volumni_best + volumni < routei.getCarriage().getTruckVolume()
										&& weight_best + weighti < routei.getCarriage().getCapacity()) {// routej重量约束
									L l = new L();
									l.setOverlapIdx(x);
									l.setSij(volumni + volumni_best);
									betterCombination.add(l);
									changed = true;
									if (volumni + volumni_best > volumni_currBest) {
										for (int xi = 0; xi < n; xi++) {
											xbest[xi] = x[xi];// 保存好xbest
										}
									}

								}
								iter = iter + 1;
							}
						}
						// xbest is the final one.根据xbest建立两条路径。
						if (!changed)
							continue;
						n_3DPacking++;//check for new_routei and new_routej
						Collections.sort(betterCombination);// 从大到小。
						for (int bc = 0; bc < Math.min(NUM_COMBINATION_MOVE_BOXES, betterCombination.size()); bc++) {
							ArrayList<Integer> currx = betterCombination.get(bc).getOverlapIdx();
							Route new_routei = new Route(routei);
							Route new_routej = new Route(routej);
							// 将nodej的箱子重新分配到new_nodei和new_nodej中。
							Node new_nodei = new Node(nodej);//for nodei
							Node new_nodej = new Node(nodej);//for nodej
							new_nodei.setGoodsNum(0);
							new_nodei.setGoodsVolumn(0.0);
							new_nodei.setGoodsWeight(0.0);
							new_nodei.setGoods(new ArrayList<Box>());
							new_nodej.setGoodsNum(0);
							new_nodej.setGoodsVolumn(0.0);
							new_nodej.setGoodsWeight(0.0);
							new_nodej.setGoods(new ArrayList<Box>());
							// new_routei.setNodes(new LinkedList<Node>());
							// new_routei.getNodes().add(new Node(depot_start));
							// Route new_routej = new Route(routej);
							// new_routej.setNodes(new LinkedList<Node>());
							// new_routej.getNodes().add(new Node(depot_start));
							int goodsNumi = 0, goodsNumj = 0;
							double goodsVolumni = 0.0, goodsVolumnj = 0.0, goodsWeighti = 0.0, goodsWeightj = 0.0;
							for (int xi = 0; xi < n; xi++) {
								Box currBox = new Box(nodej.getGoods().get(xi));
								if (currx.get(xi) == 0) {
									new_nodei.getGoods().add(currBox);
									goodsNumi += 1;
									goodsVolumni += currBox.getVolume();
									goodsWeighti += currBox.getWeight();
								} else {
									new_nodej.getGoods().add(currBox);
									goodsNumj += 1;
									goodsVolumnj += currBox.getVolume();
									goodsWeightj += currBox.getWeight();
								}
							}
							new_nodei.setGoodsNum(goodsNumi);
							new_nodei.setGoodsVolumn(goodsVolumni);
							new_nodei.setGoodsWeight(goodsWeighti);
							new_nodej.setGoodsNum(goodsNumj);
							new_nodej.setGoodsVolumn(goodsVolumnj);
							new_nodej.setGoodsWeight(goodsWeightj);
							new_routej.getNodes().set(selected_node, new_nodej);
							// new_routei.getNodes().add(new Node(depot_end));
							// new_routej.getNodes().add(new Node(depot_end));
							/**
							 * 检查是否有相同的平台在同一个路径里面。
							 */
							/**
							 * 检查是否有重复的platform 当将相同的platform的箱子换的一辆车子的时候，一般是相连的。
							 */
							// ArrayList<Integer> existPlatformID = new ArrayList<Integer>();
							int insert_pos = -1;
							for (int ni = 1; ni < new_routei.getNodes().size() - 1; ni++) {
								Node node = new_routei.getNodes().get(ni);
								if (node.getPlatformID() == nodej.getPlatformID()) {
									insert_pos = ni;
									break;
								}
							}
							if (insert_pos > 0) {
								// 已经有这个platform的箱子了，则加进去
								for (Box b : new_nodei.getGoods()) {
									new_routei.getNodes().get(insert_pos).getGoods().add(new Box(b));
								}
								int goodsNum = new_routei.getNodes().get(insert_pos).getGoodsNum()
										+ new_nodei.getGoodsNum();
								double goodsVolumn = new_routei.getNodes().get(insert_pos).getGoodsVolumn()
										+ new_nodei.getGoodsVolumn();
								double goodsWeight = new_routei.getNodes().get(insert_pos).getGoodsWeight()
										+ new_nodei.getGoodsWeight();
								new_routei.getNodes().get(insert_pos).setGoodsNum(goodsNum);
								new_routei.getNodes().get(insert_pos).setGoodsVolumn(goodsVolumn);
								new_routei.getNodes().get(insert_pos).setGoodsWeight(goodsWeight);
							} else {
								// 如果没有
								if (new_nodei.isMustFirst())
									new_routei.getNodes().add(1, new_nodei);
								else
									new_routei.getNodes().add(2, new_nodei);// 不要加载到1，以防止1是mustfist节点。而且必须确保new_nodei不是mustfirst的。
							}
							// 构建两条新的路径new_routei和new_routej结束。

							ArrayList<L> optRoutesi = findOptN(new_routei);
							int feasiblei = -1;
							boolean routei_feasible = false;
							checkidx = 0;
							while (checkidx < optRoutesi.size()) {
								ArrayList<Box> boxes = new ArrayList<Box>();
								for (int ni : optRoutesi.get(checkidx).getOverlapIdx()) {
									Node curr_node = new_routei.getNodes().get(ni);
									for (Box b : curr_node.getGoods())
										boxes.add(b);
								}
								ArrayList<Integer> k = new_routei.is_loadable(boxes);
								if (k.size() == boxes.size()) {
									for(int bi=0;bi<new_routei.getBoxes().size();bi++) {
										Box b = new_routei.getBoxes().get(bi);
										if(b.getXCoor()+b.getWidth()>new_routei.getCarriage().getWidth()) {
											System.out.println("this is not possible.");
											k = new_routei.is_loadable(boxes);
										}
									}
									routei_feasible = true;
									feasiblei = checkidx;
									break;
								}
								checkidx++;
							}
							if (!routei_feasible)
								continue;// 路径一不可行。
							// check for route i over
							// check for route j begin
							boolean routej_feasible = false;
							// ArrayList<L> optRoutesj = findOptN(new_routej,distanceMap);
							// checkidx =0;
							// while(checkidx<optRoutesj.size()) {
							ArrayList<Box> boxes = new ArrayList<Box>();
							// double weight_need = 0.0,volumn_need = 0.0;
							for (Node curr_node : new_routej.getNodes()) {
								for (Box b : curr_node.getGoods()) {
									boxes.add(b);
									// weight_need+=b.getWeight();
									// volumn_need+=b.getVolume();
								}
							}
							// for(int v=0;v<TRUCKTYPE_NUM;v++) {
							// if(BASIC_TRUCKS.get(v).getCapacity()>weight_need&&
							// BASIC_TRUCKS.get(v).getTruckVolume()>volumn_need&&
							// (BASIC_TRUCKS.get(v).getCapacity()<routej.getCarriage().getCapacity()||
							// BASIC_TRUCKS.get(v).getTruckVolume()<=routej.getCarriage().getTruckVolume())){
							// new_routej.setCarriage(BASIC_TRUCKS.get(v));
							ArrayList<Integer> k = new_routej.is_loadable(boxes);
							if (k.size() == boxes.size()) {
								routej_feasible = true;
							}
							// }
							// }
							// if(routej_feasible) break;
							// checkidx ++;
							// }
							if (routei_feasible && routej_feasible) {
								// GENERATE A NEW SOLUTION
								// allocate boxes to node in routei and routej
								LinkedList<Node> finalnodes = new LinkedList<Node>();
								finalnodes.add(new Node(depot_start));
								for (int ni : optRoutesi.get(feasiblei).getOverlapIdx()) {
									finalnodes.add(new Node(new_routei.getNodes().get(ni)));
								}
								finalnodes.add(new Node(depot_end));
								new_routei.setNodes(finalnodes);
								allocateBoxes2Node(new_routei);
								allocateBoxes2Node(new_routej);
								solution.getRoutes().set(ri, new_routei);
								solution.getRoutes().set(rj, new_routej);
								is_succ = true;
								break;
							} //
						} // for combination
						if (is_succ)
							break;
					} // for each node.
					if (is_succ)
						break;
				} // for rj
				if (is_succ)
					break;
			} // for ri
			if (is_succ) {
				solution.evaluation();
				solutionSet.add(solution);
				if (debug)
					System.out.println("after move boxes: f1: " + solution.getF1() + ";f2:" + solution.getF2());
			}
			if (n_3DPacking>=PACKING_LIMITS || (System.nanoTime() - begintime) / (1e9) > FINAL_TIME)
				break;
		}
	}// end function

	static void move_nodes(SolutionSet_vrp solutionSet) {
		// SolutionSet offspring = solutionSet;
		/**
		 * move between two route with different vehicle, move large vehicle to small
		 * vehicle. 如果两条路径有相同的节点，则会自动combine。
		 */
		double[] before = new double[2];
		for (int si = 0; si < solutionSet.size(); si++) {
			// System.out.println(si);
			Solution_vrp solution = new Solution_vrp(solutionSet.get(si));
			if (debug)
				System.out.println("before*f1: " + solution.getF1() + ";f2:" + solution.getF2());
			before[0] = solution.getF1();
			before[1] = solution.getF2();
			Route routei, routej;
			boolean is_succ = false;
			for (int ri = 0; ri < solution.getRoutes().size(); ri++) {
				routei = solution.getRoutes().get(ri);// routei是小车。
				for (int rj = 0; rj < solution.getRoutes().size(); rj++) {// 找到大车
					routej = solution.getRoutes().get(rj);// routej是大车。
					if (ri == rj || routej.getCarriage().getCapacity() <= routei.getCarriage().getCapacity()) {
						continue;
					}
					/**
					 * 如果有相同的平台，可以试着移动数量，或者后面有针对性的操作。
					 */
					// ArrayList<L> overlap = RoutesOverlap(routei,routej);
					// ArrayList<Integer> overlapPID = isOverlap(routei,routej);
					ArrayList<Node> nodes = new ArrayList<Node>();// 所有nodei+nodej
					int n = routei.getNodes().size() - 2 + routej.getNodes().size() - 2;// -overlapPID.size();
					int[] x = new int[n];
					int[] xbest = new int[n];
					int xi = 0;
					double volumni_best = 0.0;
					double volumni = 0.0, volumnj = 0.0, weighti = 0.0, weightj = 0.0;

					for (int ni = 1; ni < routei.getNodes().size() - 1; ni++) {
						nodes.add(routei.getNodes().get(ni));
						xbest[xi++] = 0;
						volumni_best += routei.getNodes().get(ni).getGoodsVolumn();
					}

					for (int nj = 1; nj < routej.getNodes().size() - 1; nj++) {
						// 这里我们允许有相同平台的箱子，把他们看成是不同平台的箱子。
						// 这是因为相同平台的箱子不可以放在一个车子里面。
						// if(overlapPID.contains(routej.getNodes().get(nj).getPlatformID())) {
						// //有相同的节点。则在已有的node里面找到
						// for(int estni=0;estni<nodes.size();estni++) {
						// Node existnode = nodes.get(estni);
						// if(existnode.getPlatformID()==routej.getNodes().get(nj).getPlatformID()) {
						// int GoodsNum =existnode.getGoodsNum();
						// double GoodsVolumn = existnode.getGoodsVolumn();
						// double GoodsWeight = existnode.getGoodsWeight();
						//// for(int bi=0;bi<routej.getNodes().get(nj).getGoods().size();bi++) {
						//// Box b = new Box(routej.getNodes().get(nj).getGoods().get(bi));
						// for(Box b:routej.getNodes().get(nj).getGoods()) {
						// existnode.getGoods().add(new Box(b));
						// GoodsNum +=1;
						// GoodsWeight += b.getWeight();
						// GoodsVolumn += b.getVolume();
						// }
						// existnode.setGoodsNum(GoodsNum);
						// existnode.setGoodsVolumn(GoodsVolumn);
						// existnode.setGoodsWeight(GoodsWeight);
						// }
						// }
						//// xbest[xi++] = 1;
						// }else {
						nodes.add(routej.getNodes().get(nj));
						xbest[xi++] = 1;
						// }
					}
					ArrayList<L> betterCombination = new ArrayList<L>();
					boolean changed = false;
					boolean feasible_flag = true;
					if (n <= MOVE_NODE_BRUTEFORCE) {// 全部遍历。
						int[][] allx = new int[(int) Math.pow(2, n)][n];

						for (int i = 0; i < Math.pow(2, n); i++) {
							String curr = Integer.toBinaryString(i);
							int char_idx = curr.length() - 1;
							volumni = 0.0;
							weighti = 0.0;
							volumnj = 0.0;
							weightj = 0.0;
							feasible_flag = true;
							for (int idx = n - 1; idx >= 0; idx--) {
								if (char_idx >= 0) {
									allx[i][idx] = Character.getNumericValue(curr.charAt(char_idx));
									char_idx--;
								} else
									allx[i][idx] = 0;
								if (allx[i][idx] == 0 && isLoadable(routei.getCarriage(), nodes.get(idx).getGoods())) {// node
																														// idx
																														// is
																														// load
																														// by
																														// routei
									volumni += nodes.get(idx).getGoodsVolumn();
									weighti += nodes.get(idx).getGoodsWeight();
								} else if (allx[i][idx] == 1
										&& isLoadable(routej.getCarriage(), nodes.get(idx).getGoods())) {
									volumnj += nodes.get(idx).getGoodsVolumn();//
									weightj += nodes.get(idx).getGoodsWeight();
								} else {
									feasible_flag = false;
									// System.out.println("why this node can not be loaded??");
								}
							}
							if (feasible_flag && volumni > volumni_best
									&& volumni < routei.getCarriage().getTruckVolume()
									&& weighti < routei.getCarriage().getCapacity() && // routei重量约束
									weightj < routej.getCarriage().getCapacity()) {// routej重量约束
								L l = new L();
								l.setOverlapIdx(allx[i]);
								l.setSij(volumni - volumni_best);
								betterCombination.add(l);
								changed = true;
							}
						}
					} else {
						if (debug)
							System.out.println("this should be revised.");
						// 用进化算法来求解一个组合。(bit-flip)
						// int iter=0;
						// int randi = rand.nextInt(n-1);
						// while(iter<100) {
						// randi = rand.nextInt(n-1);
						// //disturb to generate a new solution
						// volumni = 0.0;
						// volumnj=0.0;
						// for(xi=0;xi<n;xi++) {
						// if(rand.nextDouble()<0.2||xi==randi) {//keep the best.
						// if(rand.nextDouble()<0.5) {
						// x[xi]=0;volumni+=nodes.get(xi).getGoodsVolumn();// node xi is selected.
						// weighti+=nodes.get(xi).getGoodsWeight();
						// }
						// else {
						// x[xi]=1;volumnj+=nodes.get(xi).getGoodsVolumn();//
						// weightj+=nodes.get(xi).getGoodsWeight();
						// }
						// }else {//disturb with probability 0.2
						// x[xi]=xbest[xi];
						// if(x[xi]==0) {
						// volumni+=nodes.get(xi).getGoodsVolumn();
						// weighti+=nodes.get(xi).getGoodsWeight();
						// }else {
						// volumnj+=nodes.get(xi).getGoodsVolumn();
						// weightj+=nodes.get(xi).getGoodsWeight();
						// }
						// }
						// }
						// //select
						// if(volumni>volumni_best&&volumni<routei.getCarriage().getTruckVolume()&&
						// weighti<routei.getCarriage().getCapacity()&&//routei重量约束
						// weightj<routej.getCarriage().getCapacity()) {//routej重量约束
						// for(xi=0;xi<n;xi++) xbest[xi]=x[xi];
						// changed = true;
						// }
						// }
					}
					// xbest is the final one.根据xbest建立两条路径。
					if (!changed)
						continue;
					n_3DPacking++;//check for new_routei and new_routej
					Collections.sort(betterCombination);// 从大到小。
					// System.out.println("check better combination:"+betterCombination.size());
					for (int bc = 0; bc < Math.min(NUM_COMBINATION_MOVE_NODES, betterCombination.size()); bc++) {
						ArrayList<Integer> currx = betterCombination.get(bc).getOverlapIdx();
						Route new_routei = new Route(routei);
						new_routei.setNodes(new LinkedList<Node>());
						new_routei.getNodes().add(new Node(routei.getNodes().get(0)));
						Route new_routej = new Route(routej);
						new_routej.setNodes(new LinkedList<Node>());
						new_routej.getNodes().add(new Node(routej.getNodes().get(0)));
						for (xi = 0; xi < n; xi++) {
							if (currx.get(xi) == 0) {
								if (nodes.get(xi).isMustFirst())
									new_routei.getNodes().add(1, new Node(nodes.get(xi)));
								else
									new_routei.getNodes().add(new Node(nodes.get(xi)));
							} else {
								if (nodes.get(xi).isMustFirst())
									new_routej.getNodes().add(1, new Node(nodes.get(xi)));
								else
									new_routej.getNodes().add(new Node(nodes.get(xi)));
							}
						}
						new_routei.getNodes().add(new Node(routei.getNodes().getLast()));
						new_routej.getNodes().add(new Node(routej.getNodes().getLast()));
						/**
						 * 检查是否有相同的平台在同一个路径里面。
						 */
						/**
						 * 检查是否有重复的platform 当将相同的platform的箱子换的一辆车子的时候，一般是相连的。
						 */
						ArrayList<Integer> existPlatformID = new ArrayList<Integer>();
						for (int ni = 1; ni < new_routei.getNodes().size() - 1; ni++) {
							Node node = new_routei.getNodes().get(ni);
							if (existPlatformID.contains(node.getPlatformID())) {
								if (node.isMustFirst()
										|| node.getPlatformID() == new_routei.getNodes().get(ni - 1).getPlatformID()) {
									Node existNode;
									if (node.isMustFirst())
										existNode = new_routei.getNodes().get(1);
									else
										existNode = new_routei.getNodes().get(ni - 1);

									for (Box b : node.getGoods()) {
										existNode.getGoods().add(new Box(b));
									}
									existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
									existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
									existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
									new_routei.getNodes().remove(ni);
								}
								// else
								// if(ni+1<currRoute.getNodes().size()-1&&node.getPlatformID()==currRoute.getNodes().get(ni+1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }
								// if(node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID()) {
								// currRoute.getNodes().remove(ni);
								// }
								else {
									// if(debug)
									// System.out.println("two same platform in a route without connected.!!!!!");

									int existid;
									for (existid = 1; existid < ni; existid++) {
										if (new_routei.getNodes().get(existid).getPlatformID() == node
												.getPlatformID()) {
											break;
										}
									}
									String existPID_1 = String
											.valueOf(new_routei.getNodes().get(existid - 1).getPlatformID());
									String existPID0 = String
											.valueOf(new_routei.getNodes().get(existid).getPlatformID());
									String existPID1 = String
											.valueOf(new_routei.getNodes().get(existid + 1).getPlatformID());
									String currPID_1 = String
											.valueOf(new_routei.getNodes().get(ni - 1).getPlatformID());
									String currPID0 = String.valueOf(new_routei.getNodes().get(ni).getPlatformID());
									String currPID1 = String.valueOf(new_routei.getNodes().get(ni + 1).getPlatformID());
									double distanceExsit = distanceMap.get(existPID_1 + "+" + existPID0)
											+ distanceMap.get(existPID0 + "+" + existPID1)
											- distanceMap.get(existPID_1 + "+" + existPID1);
									double distanceNow = distanceMap.get(currPID_1 + "+" + currPID0)
											+ distanceMap.get(currPID0 + "+" + currPID1)
											- distanceMap.get(currPID_1 + "+" + currPID1);
									// 哪个距离短旧删除哪个。
									if (distanceExsit > distanceNow) {
										// 组合到exist
										Node existNode = new_routei.getNodes().get(existid);
										for (Box b : node.getGoods()) {
											existNode.getGoods().add(new Box(b));
										}
										existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
										existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
										existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
										new_routei.getNodes().remove(ni);
									} else {
										// 组合到distanceNow
										Node existNode = new_routei.getNodes().get(existid);
										for (Box b : existNode.getGoods()) {
											node.getGoods().add(new Box(b));
										}
										node.setGoodsNum(node.getGoodsNum() + existNode.getGoodsNum());
										node.setGoodsWeight(node.getGoodsWeight() + existNode.getGoodsWeight());
										node.setGoodsVolumn(node.getGoodsVolumn() + existNode.getGoodsVolumn());
										new_routei.getNodes().remove(existid);
									}
								}
								ni = ni - 1;// 必须要向前移动一个。。。
							} else {
								existPlatformID.add(node.getPlatformID());
							}
						}
						// check for route i
						boolean routei_feasible = false;
						ArrayList<L> optRoutesi = findOptN(new_routei);
						int checkidx = 0;
						int feasiblei = -1, feasiblej = -1;
						while (checkidx < optRoutesi.size()) {
							ArrayList<Box> boxes = new ArrayList<Box>();
							for (int ni : optRoutesi.get(checkidx).getOverlapIdx()) {
								Node curr_node = new_routei.getNodes().get(ni);
								for (Box b : curr_node.getGoods())
									boxes.add(b);
							}
							ArrayList<Integer> k = new_routei.is_loadable(boxes);
							if (k.size() == boxes.size()) {
								routei_feasible = true;
								feasiblei = checkidx;
								break;
							}
							checkidx++;
						}
						if (!routei_feasible)
							continue;// 路径一不可行。
						// check for route j
						/**
						 * 检查是否有相同的平台在同一个路径里面。
						 */
						/**
						 * 检查是否有重复的platform 当将相同的platform的箱子换的一辆车子的时候，一般是相连的。
						 */
						existPlatformID = new ArrayList<Integer>();
						for (int ni = 1; ni < new_routej.getNodes().size() - 1; ni++) {
							Node node = new_routej.getNodes().get(ni);
							if (existPlatformID.contains(node.getPlatformID())) {
								if (node.getPlatformID() == new_routej.getNodes().get(ni - 1).getPlatformID()) {
									Node existNode = new_routej.getNodes().get(ni - 1);
									for (Box b : node.getGoods()) {
										existNode.getGoods().add(new Box(b));
									}
									existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
									existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
									existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
									new_routej.getNodes().remove(ni);
								}
								// if(ni-1>=1&&node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }else
								// if(ni+1<currRoute.getNodes().size()-1&&node.getPlatformID()==currRoute.getNodes().get(ni+1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }
								// if(node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID()) {
								// currRoute.getNodes().remove(ni);
								// }
								else {
									if (debug)
										System.out.println("two same platform in a route without connected.!!!!!");
									//
									int existid;
									for (existid = 1; existid < ni; existid++) {
										if (new_routej.getNodes().get(existid).getPlatformID() == node
												.getPlatformID()) {
											break;
										}
									}
									String existPID_1 = String
											.valueOf(new_routej.getNodes().get(existid - 1).getPlatformID());
									String existPID0 = String
											.valueOf(new_routej.getNodes().get(existid).getPlatformID());
									String existPID1 = String
											.valueOf(new_routej.getNodes().get(existid + 1).getPlatformID());
									String currPID_1 = String
											.valueOf(new_routej.getNodes().get(ni - 1).getPlatformID());
									String currPID0 = String.valueOf(new_routej.getNodes().get(ni).getPlatformID());
									String currPID1 = String.valueOf(new_routej.getNodes().get(ni + 1).getPlatformID());
									double distanceExsit = distanceMap.get(existPID_1 + "+" + existPID0)
											+ distanceMap.get(existPID0 + "+" + existPID1)
											- distanceMap.get(existPID_1 + "+" + existPID1);
									double distanceNow = distanceMap.get(currPID_1 + "+" + currPID0)
											+ distanceMap.get(currPID0 + "+" + currPID1)
											- distanceMap.get(currPID_1 + "+" + currPID1);
									// 哪个距离短旧删除哪个。
									if (distanceExsit > distanceNow) {
										// 组合到exist
										Node existNode = new_routej.getNodes().get(existid);
										for (Box b : node.getGoods()) {
											existNode.getGoods().add(new Box(b));
										}
										existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
										existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
										existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
										new_routej.getNodes().remove(ni);
									} else {// if(distanceExsit<distanceNow)还有相等的情况。
										// 组合到distanceNow
										Node existNode = new_routej.getNodes().get(existid);
										for (Box b : existNode.getGoods()) {
											node.getGoods().add(new Box(b));
										}
										node.setGoodsNum(node.getGoodsNum() + existNode.getGoodsNum());
										node.setGoodsWeight(node.getGoodsWeight() + existNode.getGoodsWeight());
										node.setGoodsVolumn(node.getGoodsVolumn() + existNode.getGoodsVolumn());
										new_routej.getNodes().remove(existid);
									}
								}
								ni = ni - 1;// 必须要向前移动一个。。。
							} else {
								existPlatformID.add(node.getPlatformID());
							}
						}
						boolean routej_feasible = false;
						ArrayList<L> optRoutesj = findOptN(new_routej);
						checkidx = 0;
						while (checkidx < optRoutesj.size()) {
							ArrayList<Box> boxes = new ArrayList<Box>();
							// double weight_need = 0.0,volumn_need = 0.0;
							for (int nj : optRoutesj.get(checkidx).getOverlapIdx()) {
								Node curr_node = new_routej.getNodes().get(nj);
								for (Box b : curr_node.getGoods()) {
									boxes.add(b);
									// weight_need+=b.getWeight();
									// volumn_need+=b.getVolume();
								}
							}
							// for(int v=0;v<TRUCKTYPE_NUM;v++) {
							// if(BASIC_TRUCKS.get(v).getCapacity()>weight_need&&
							// BASIC_TRUCKS.get(v).getTruckVolume()>volumn_need&&
							// (BASIC_TRUCKS.get(v).getCapacity()<routej.getCarriage().getCapacity()||
							// BASIC_TRUCKS.get(v).getTruckVolume()<=routej.getCarriage().getTruckVolume())){
							// new_routej.setCarriage(BASIC_TRUCKS.get(v));
							ArrayList<Integer> k = new_routej.is_loadable(boxes);
							if (k.size() == boxes.size()) {
								routej_feasible = true;
								feasiblej = checkidx;
								break;
							}
							// }
							// }
							if (routej_feasible)
								break;
							checkidx++;
						}
						if (routei_feasible && routej_feasible) {
							// GENERATE A NEW SOLUTION
							// allocate boxes to node in routei and routej
							LinkedList<Node> finalnodes = new LinkedList<Node>();
							finalnodes.add(new Node(depot_start));
							for (int ni : optRoutesi.get(feasiblei).getOverlapIdx()) {
								finalnodes.add(new Node(new_routei.getNodes().get(ni)));
							}
							finalnodes.add(new Node(depot_end));
							new_routei.setNodes(finalnodes);
							allocateBoxes2Node(new_routei);
							finalnodes = new LinkedList<Node>();
							finalnodes.add(new Node(depot_start));
							for (int nj : optRoutesj.get(feasiblej).getOverlapIdx()) {
								finalnodes.add(new Node(new_routej.getNodes().get(nj)));
							}
							finalnodes.add(new Node(depot_end));
							new_routej.setNodes(finalnodes);
							allocateBoxes2Node(new_routej);
							solution.getRoutes().set(ri, new_routei);
							solution.getRoutes().set(rj, new_routej);
							is_succ = true;
							break;
						} //
					} // for combination
					if (is_succ)
						break;
				} // for rj
				if (is_succ)
					break;
			} // for ri
			if (is_succ) {
				solution.evaluation();
				// if(solution.getF1()==before[0]&&solution.getF2()==before[1])
				// System.out.println("a same solution.....");
				if (solution.getF1() < before[0] || solution.getF2() < before[1]) {
					// System.out.println("a better solution.");
					solutionSet.add(solution);
					if (debug)
						System.out.println("after*f1: " + solution.getF1() + ";f2:" + solution.getF2());
				}
			}
			if (n_3DPacking>=PACKING_LIMITS || (System.nanoTime() - begintime) / (1e9) > FINAL_TIME)
				break;
		} // for si
	}

	/**
	 * 在两个相同车子之间移动节点。目的使得距离更短，且可行。
	 * 
	 * @param solutionSet
	 * @param distanceMap
	 */
	static void move_nodes_same(SolutionSet_vrp solutionSet) {
		// SolutionSet offspring = solutionSet;
		/**
		 * move between two route with different vehicle, move large vehicle to small
		 * vehicle. 如果两条路径有相同的节点，则会自动combine。
		 */

		for (int si = 0; si < solutionSet.size(); si++) {
			Solution_vrp solution = new Solution_vrp(solutionSet.get(si));
			// System.out.println(si+"/"+solutionSet.size());
			if (debug)
				System.out.println("before*f1: " + solution.getF1() + ";f2:" + solution.getF2());
			double[] before = { solution.getF1(), solution.getF2() };
			Route routei, routej;
			boolean is_succ = false;
			for (int ri = 0; ri < solution.getRoutes().size(); ri++) {
				routei = solution.getRoutes().get(ri);// routei是小车。
				for (int rj = ri + 1; rj < solution.getRoutes().size(); rj++) {
					routej = solution.getRoutes().get(rj);// routej
					if (routej.getCarriage().getTruckTypeCode() != routei.getCarriage().getTruckTypeCode()) {
						continue;
					}
					/**
					 * 如果有相同的平台，可以试着移动数量，或者后面有针对性的操作。
					 */
					// ArrayList<L> overlap = RoutesOverlap(routei,routej);
					// ArrayList<Integer> overlapPID = isOverlap(routei,routej);
					ArrayList<Node> nodes = new ArrayList<Node>();// 所有nodei+nodej
					int n = routei.getNodes().size() - 2 + routej.getNodes().size() - 2;// -overlapPID.size();
					int[] x = new int[n];
					int[] xbest = new int[n];
					int xi = 0;
					double dist_best = calculateDist(routei.getNodes())
							+ calculateDist(routej.getNodes());
					double volumni = 0.0, volumnj = 0.0, weighti = 0.0, weightj = 0.0;
					int existi = 0;
					for (int ni = 1; ni < routei.getNodes().size() - 1; ni++) {
						nodes.add(routei.getNodes().get(ni));
						xbest[xi++] = 0;
						// volumni_best+=routei.getNodes().get(ni).getGoodsVolumn();
					}

					for (int nj = 1; nj < routej.getNodes().size() - 1; nj++) {
						// 这里我们允许有相同平台的箱子，把他们看成是不同平台的箱子。
						// 这是因为相同平台的箱子不可以放在一个车子里面。
						// if(overlapPID.contains(routej.getNodes().get(nj).getPlatformID())) {
						// //有相同的节点。则在已有的node里面找到
						// for(int estni=0;estni<nodes.size();estni++) {
						// Node existnode = nodes.get(estni);
						// if(existnode.getPlatformID()==routej.getNodes().get(nj).getPlatformID()) {
						// int GoodsNum =existnode.getGoodsNum();
						// double GoodsVolumn = existnode.getGoodsVolumn();
						// double GoodsWeight = existnode.getGoodsWeight();
						//// for(int bi=0;bi<routej.getNodes().get(nj).getGoods().size();bi++) {
						//// Box b = new Box(routej.getNodes().get(nj).getGoods().get(bi));
						// for(Box b:routej.getNodes().get(nj).getGoods()) {
						// existnode.getGoods().add(new Box(b));
						// GoodsNum +=1;
						// GoodsWeight += b.getWeight();
						// GoodsVolumn += b.getVolume();
						// }
						// existnode.setGoodsNum(GoodsNum);
						// existnode.setGoodsVolumn(GoodsVolumn);
						// existnode.setGoodsWeight(GoodsWeight);
						// }
						// }
						//// xbest[xi++] = 1;
						// }else {
						nodes.add(routej.getNodes().get(nj));
						existi = existi + (int) Math.pow(2, n - 1 - xi);
						xbest[xi++] = 1;

						// }
					}
					ArrayList<L> betterCombination = new ArrayList<L>();
					boolean changed = false;
					boolean feasible_flag = true;
					LinkedList<Node> nodesi = new LinkedList<Node>();
					LinkedList<Node> nodesj = new LinkedList<Node>();

					if (n <= MOVE_NODE_BRUTEFORCE) {// 全部遍历。
						int[][] allx = new int[(int) Math.pow(2, n)][n];
						int mustfirst = 0;
						for (int i = 0; i < Math.pow(2, n) && i != existi; i++) {
							String curr = Integer.toBinaryString(i);
							int char_idx = curr.length() - 1;
							volumni = 0.0;
							weighti = 0.0;
							volumnj = 0.0;
							weightj = 0.0;
							feasible_flag = true;
							nodesi = new LinkedList<Node>();
							nodesi.add(routei.getNodes().getFirst());
							nodesj = new LinkedList<Node>();
							nodesj.add(routej.getNodes().getFirst());
							mustfirst = 0;
							for (int idx = n - 1; idx >= 0; idx--) {
								if (char_idx >= 0) {
									allx[i][idx] = Character.getNumericValue(curr.charAt(char_idx));
									char_idx--;
								} else
									allx[i][idx] = 0;
								if (allx[i][idx] == 0 && isLoadable(routei.getCarriage(), nodes.get(idx).getGoods())) {// node
																														// idx
																														// is
																														// load
																														// by
																														// routei
									volumni += nodes.get(idx).getGoodsVolumn();
									weighti += nodes.get(idx).getGoodsWeight();
									if (nodes.get(idx).isMustFirst()) {
										nodesi.add(1, nodes.get(idx));
										mustfirst++;
									} else
										nodesi.add(nodes.get(idx));
								} else if (allx[i][idx] == 1
										&& isLoadable(routej.getCarriage(), nodes.get(idx).getGoods())) {
									volumnj += nodes.get(idx).getGoodsVolumn();//
									weightj += nodes.get(idx).getGoodsWeight();
									if (nodes.get(idx).isMustFirst()) {
										nodesj.add(1, nodes.get(idx));
										mustfirst++;
									} else
										nodesj.add(nodes.get(idx));
								} else {
									feasible_flag = false;
									// System.out.println("why this node can not be loaded??");
								}
							}

							if (mustfirst <= 1 && feasible_flag && volumni < routei.getCarriage().getTruckVolume()
									&& volumnj < routej.getCarriage().getTruckVolume()
									&& weighti < routei.getCarriage().getCapacity() && // routei重量约束
									weightj < routej.getCarriage().getCapacity()) {// routej重量约束
								nodesi.add(routei.getNodes().getLast());
								nodesj.add(routej.getNodes().getLast());
								double dist_curr = bestVRP(nodesi) + bestVRP(nodesj);
								if (dist_curr < dist_best) {
									L l = new L();
									l.setOverlapIdx(allx[i]);
									l.setSij(dist_curr);
									betterCombination.add(l);
									changed = true;
								}
							}
						}
					} else {
						if (debug)
							System.out.println("this should be revised.");
						// 用进化算法来求解一个组合。(bit-flip)
						// int iter=0;
						// int randi = rand.nextInt(n-1);
						// while(iter<100) {
						// randi = rand.nextInt(n-1);
						// //disturb to generate a new solution
						// volumni = 0.0;
						// volumnj=0.0;
						// for(xi=0;xi<n;xi++) {
						// if(rand.nextDouble()<0.2||xi==randi) {//keep the best.
						// if(rand.nextDouble()<0.5) {
						// x[xi]=0;volumni+=nodes.get(xi).getGoodsVolumn();// node xi is selected.
						// weighti+=nodes.get(xi).getGoodsWeight();
						// }
						// else {
						// x[xi]=1;volumnj+=nodes.get(xi).getGoodsVolumn();//
						// weightj+=nodes.get(xi).getGoodsWeight();
						// }
						// }else {//disturb with probability 0.2
						// x[xi]=xbest[xi];
						// if(x[xi]==0) {
						// volumni+=nodes.get(xi).getGoodsVolumn();
						// weighti+=nodes.get(xi).getGoodsWeight();
						// }else {
						// volumnj+=nodes.get(xi).getGoodsVolumn();
						// weightj+=nodes.get(xi).getGoodsWeight();
						// }
						// }
						// }
						// //select
						// if(volumni>volumni_best&&volumni<routei.getCarriage().getTruckVolume()&&
						// weighti<routei.getCarriage().getCapacity()&&//routei重量约束
						// weightj<routej.getCarriage().getCapacity()) {//routej重量约束
						// for(xi=0;xi<n;xi++) xbest[xi]=x[xi];
						// changed = true;
						// }
						// }
					}
					// xbest is the final one.根据xbest建立两条路径。
					if (!changed)
						continue;
					Collections.sort(betterCombination);// 从大到小。
					// System.out.println(betterCombination.size());
					for (int bc = 0; bc < Math.min(NUM_COMBINATION_MOVE_NODES, betterCombination.size()); bc++) {
						ArrayList<Integer> currx = betterCombination.get(bc).getOverlapIdx();
						Route new_routei = new Route(routei);
						new_routei.setNodes(new LinkedList<Node>());
						new_routei.getNodes().add(new Node(depot_start));
						Route new_routej = new Route(routej);
						new_routej.setNodes(new LinkedList<Node>());
						new_routej.getNodes().add(new Node(depot_start));
						for (xi = 0; xi < n; xi++) {
							if (currx.get(xi) == 0) {
								if (nodes.get(xi).isMustFirst())
									new_routei.getNodes().add(1, new Node(nodes.get(xi)));
								else
									new_routei.getNodes().add(new Node(nodes.get(xi)));
							} else {
								if (nodes.get(xi).isMustFirst())
									new_routej.getNodes().add(1, new Node(nodes.get(xi)));
								else
									new_routej.getNodes().add(new Node(nodes.get(xi)));
							}
						}
						new_routei.getNodes().add(new Node(depot_end));
						new_routej.getNodes().add(new Node(depot_end));
						/**
						 * 检查是否有相同的平台在同一个路径里面。
						 */
						/**
						 * 检查是否有重复的platform 当将相同的platform的箱子换的一辆车子的时候，一般是相连的。
						 */
						ArrayList<Integer> existPlatformID = new ArrayList<Integer>();
						for (int ni = 1; ni < new_routei.getNodes().size() - 1; ni++) {
							Node node = new_routei.getNodes().get(ni);
							if (existPlatformID.contains(node.getPlatformID())) {
								if (node.isMustFirst()
										|| node.getPlatformID() == new_routei.getNodes().get(ni - 1).getPlatformID()) {
									// Node existNode = new_routei.getNodes().get(ni-1);
									Node existNode;
									if (node.isMustFirst())
										existNode = new_routei.getNodes().get(1);
									else
										existNode = new_routei.getNodes().get(ni - 1);
									for (Box b : node.getGoods()) {
										existNode.getGoods().add(new Box(b));
									}
									existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
									existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
									existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
									new_routei.getNodes().remove(ni);
								}
								// else
								// if(ni+1<currRoute.getNodes().size()-1&&node.getPlatformID()==currRoute.getNodes().get(ni+1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }
								// if(node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID()) {
								// currRoute.getNodes().remove(ni);
								// }
								else {
									// if(debug)
									// System.out.println("two same platform in a route without connected.!!!!!");

									int existid;
									for (existid = 1; existid < ni; existid++) {
										if (new_routei.getNodes().get(existid).getPlatformID() == node
												.getPlatformID()) {
											break;
										}
									}
									String existPID_1 = String
											.valueOf(new_routei.getNodes().get(existid - 1).getPlatformID());
									String existPID0 = String
											.valueOf(new_routei.getNodes().get(existid).getPlatformID());
									String existPID1 = String
											.valueOf(new_routei.getNodes().get(existid + 1).getPlatformID());
									String currPID_1 = String
											.valueOf(new_routei.getNodes().get(ni - 1).getPlatformID());
									String currPID0 = String.valueOf(new_routei.getNodes().get(ni).getPlatformID());
									String currPID1 = String.valueOf(new_routei.getNodes().get(ni + 1).getPlatformID());
									double distanceExsit = distanceMap.get(existPID_1 + "+" + existPID0)
											+ distanceMap.get(existPID0 + "+" + existPID1)
											- distanceMap.get(existPID_1 + "+" + existPID1);
									double distanceNow = distanceMap.get(currPID_1 + "+" + currPID0)
											+ distanceMap.get(currPID0 + "+" + currPID1)
											- distanceMap.get(currPID_1 + "+" + currPID1);
									// 哪个距离短旧删除哪个。
									if (distanceExsit > distanceNow) {
										// 组合到exist
										Node existNode = new_routei.getNodes().get(existid);
										for (Box b : node.getGoods()) {
											existNode.getGoods().add(new Box(b));
										}
										existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
										existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
										existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
										new_routei.getNodes().remove(ni);
									} else {
										// 组合到distanceNow
										Node existNode = new_routei.getNodes().get(existid);
										for (Box b : existNode.getGoods()) {
											node.getGoods().add(new Box(b));
										}
										node.setGoodsNum(node.getGoodsNum() + existNode.getGoodsNum());
										node.setGoodsWeight(node.getGoodsWeight() + existNode.getGoodsWeight());
										node.setGoodsVolumn(node.getGoodsVolumn() + existNode.getGoodsVolumn());
										new_routei.getNodes().remove(existid);
									}
								}
								ni = ni - 1;// 必须要向前移动一个。。。
							} else {
								existPlatformID.add(node.getPlatformID());
							}
						}
						// check for route i
						boolean routei_feasible = false;
						ArrayList<L> optRoutesi = findOptN(new_routei);
						int checkidx = 0;
						int feasiblei = -1, feasiblej = -1;
						while (checkidx < optRoutesi.size()) {
							ArrayList<Box> boxes = new ArrayList<Box>();
							for (int ni : optRoutesi.get(checkidx).getOverlapIdx()) {
								Node curr_node = new_routei.getNodes().get(ni);
								for (Box b : curr_node.getGoods())
									boxes.add(b);
							}
							ArrayList<Integer> k = new_routei.is_loadable(boxes);
							if (k.size() == boxes.size()) {
								routei_feasible = true;
								feasiblei = checkidx;
								break;
							}
							checkidx++;
						}
						if (!routei_feasible)
							continue;// 路径一不可行。
						n_3DPacking++;// check for route j
						/**
						 * 检查是否有相同的平台在同一个路径里面。
						 */
						/**
						 * 检查是否有重复的platform 当将相同的platform的箱子换的一辆车子的时候，一般是相连的。
						 */
						existPlatformID = new ArrayList<Integer>();
						for (int ni = 1; ni < new_routej.getNodes().size() - 1; ni++) {
							Node node = new_routej.getNodes().get(ni);
							if (existPlatformID.contains(node.getPlatformID())) {
								if (node.getPlatformID() == new_routej.getNodes().get(ni - 1).getPlatformID()) {
									Node existNode = new_routej.getNodes().get(ni - 1);
									for (Box b : node.getGoods()) {
										existNode.getGoods().add(new Box(b));
									}
									existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
									existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
									existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
									new_routej.getNodes().remove(ni);
								}
								// if(ni-1>=1&&node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }else
								// if(ni+1<currRoute.getNodes().size()-1&&node.getPlatformID()==currRoute.getNodes().get(ni+1).getPlatformID())
								// {
								// //combine this two nodes
								// currRoute.getNodes().remove(ni);
								// }
								// if(node.getPlatformID()==currRoute.getNodes().get(ni-1).getPlatformID()) {
								// currRoute.getNodes().remove(ni);
								// }
								else {
									if (debug)
										System.out.println("two same platform in a route without connected.!!!!!");
									//
									int existid;
									for (existid = 1; existid < ni; existid++) {
										if (new_routej.getNodes().get(existid).getPlatformID() == node
												.getPlatformID()) {
											break;
										}
									}
									String existPID_1 = String
											.valueOf(new_routej.getNodes().get(existid - 1).getPlatformID());
									String existPID0 = String
											.valueOf(new_routej.getNodes().get(existid).getPlatformID());
									String existPID1 = String
											.valueOf(new_routej.getNodes().get(existid + 1).getPlatformID());
									String currPID_1 = String
											.valueOf(new_routej.getNodes().get(ni - 1).getPlatformID());
									String currPID0 = String.valueOf(new_routej.getNodes().get(ni).getPlatformID());
									String currPID1 = String.valueOf(new_routej.getNodes().get(ni + 1).getPlatformID());
									double distanceExsit = distanceMap.get(existPID_1 + "+" + existPID0)
											+ distanceMap.get(existPID0 + "+" + existPID1)
											- distanceMap.get(existPID_1 + "+" + existPID1);
									double distanceNow = distanceMap.get(currPID_1 + "+" + currPID0)
											+ distanceMap.get(currPID0 + "+" + currPID1)
											- distanceMap.get(currPID_1 + "+" + currPID1);
									// 哪个距离短旧删除哪个。
									if (distanceExsit > distanceNow) {
										// 组合到exist
										Node existNode = new_routej.getNodes().get(existid);
										for (Box b : node.getGoods()) {
											existNode.getGoods().add(new Box(b));
										}
										existNode.setGoodsNum(existNode.getGoodsNum() + node.getGoodsNum());
										existNode.setGoodsWeight(existNode.getGoodsWeight() + node.getGoodsWeight());
										existNode.setGoodsVolumn(existNode.getGoodsVolumn() + node.getGoodsVolumn());
										new_routej.getNodes().remove(ni);
									} else {// if(distanceExsit<distanceNow)还有相等的情况。
										// 组合到distanceNow
										Node existNode = new_routej.getNodes().get(existid);
										for (Box b : existNode.getGoods()) {
											node.getGoods().add(new Box(b));
										}
										node.setGoodsNum(node.getGoodsNum() + existNode.getGoodsNum());
										node.setGoodsWeight(node.getGoodsWeight() + existNode.getGoodsWeight());
										node.setGoodsVolumn(node.getGoodsVolumn() + existNode.getGoodsVolumn());
										new_routej.getNodes().remove(existid);
									}
								}
								ni = ni - 1;// 必须要向前移动一个。。。
							} else {
								existPlatformID.add(node.getPlatformID());
							}
						}
						boolean routej_feasible = false;
						ArrayList<L> optRoutesj = findOptN(new_routej);
						checkidx = 0;
						while (checkidx < optRoutesj.size()) {
							ArrayList<Box> boxes = new ArrayList<Box>();
							// double weight_need = 0.0,volumn_need = 0.0;
							for (int nj : optRoutesj.get(checkidx).getOverlapIdx()) {
								Node curr_node = new_routej.getNodes().get(nj);
								for (Box b : curr_node.getGoods()) {
									boxes.add(b);
									// weight_need+=b.getWeight();
									// volumn_need+=b.getVolume();
								}
							}
							// for(int v=0;v<TRUCKTYPE_NUM;v++) {
							// if(BASIC_TRUCKS.get(v).getCapacity()>weight_need&&
							// BASIC_TRUCKS.get(v).getTruckVolume()>volumn_need&&
							// (BASIC_TRUCKS.get(v).getCapacity()<routej.getCarriage().getCapacity()||
							// BASIC_TRUCKS.get(v).getTruckVolume()<=routej.getCarriage().getTruckVolume())){
							// new_routej.setCarriage(BASIC_TRUCKS.get(v));
							ArrayList<Integer> k = new_routej.is_loadable(boxes);
							if (k.size() == boxes.size()) {
								routej_feasible = true;
								feasiblej = checkidx;
								break;
							}
							// }
							// }
							if (routej_feasible)
								break;
							checkidx++;
						}
						if (routei_feasible && routej_feasible) {
							// GENERATE A NEW SOLUTION
							// allocate boxes to node in routei and routej
							LinkedList<Node> finalnodes = new LinkedList<Node>();
							finalnodes.add(new Node(depot_start));
							for (int ni : optRoutesi.get(feasiblei).getOverlapIdx()) {
								finalnodes.add(new Node(new_routei.getNodes().get(ni)));
							}
							finalnodes.add(new Node(depot_end));
							new_routei.setNodes(finalnodes);
							allocateBoxes2Node(new_routei);
							finalnodes = new LinkedList<Node>();
							finalnodes.add(new Node(depot_start));
							for (int nj : optRoutesj.get(feasiblej).getOverlapIdx()) {
								finalnodes.add(new Node(new_routej.getNodes().get(nj)));
							}
							finalnodes.add(new Node(depot_end));
							new_routej.setNodes(finalnodes);
							allocateBoxes2Node(new_routej);
							solution.getRoutes().set(ri, new_routei);
							solution.getRoutes().set(rj, new_routej);
							is_succ = true;
							break;
						} //
					} // for combination
					if (is_succ)
						break;
				} // for rj
				if (is_succ)
					break;
			} // for ri
			if (is_succ) {
				solution.evaluation();
				// if(solution.getF1()==before[0]&&solution.getF2()==before[1])
				// System.out.println("A same solution is generated....");
				if (solution.getF1() < before[0] || solution.getF2() < before[1]) {
					// System.out.println("a better solution is generated.");
					solutionSet.add(solution);

				}
				// if(debug)
				// outputJSON(solutionSet, "debug.file", PlatformIDCodeMap,
				// output_directory+'/'+filenames[fileidx]);
				if (debug)
					System.out.println(
							"after move between same car: f1: " + solution.getF1() + ";f2:" + solution.getF2());
			}
			if (n_3DPacking>=PACKING_LIMITS ||(System.nanoTime() - begintime) / (1e9) > FINAL_TIME)
				break;
		}
	}
	
	static SolutionSet_vrp get_Solutions(double  split_minv, ArrayList<Solution> solutionsList_0, double relax_ratio,double[] truck_weight_ratio, int[] save_feasible_no, double[]if_hard_node) throws IOException {
		double[][] client_v_w = new double[CLIENT_NUM][2];//client_volume_weight;
	    for(int i=0;i<CLIENT_NUM;i++) {
	    	client_v_w[i][0] = client_volume_weight[i][0];
	    	client_v_w[i][1] = client_volume_weight[i][1];
	    }
	    double Split_precision = 6.0;
	    int x = 0 ;
	    
   
	    SolutionSet_vrp solutionSet = new SolutionSet_vrp();

	  
	    for (int n_sol =0; n_sol <solutionsList_0.size(); n_sol++) {
	    	n_3DPacking++;	//check for the feasibility of this solution.
	        if(n_3DPacking>PACKING_LIMITS) {
	        	break;
	        }
	    	Solution aSolutionsList_ = solutionsList_0.get(n_sol);
	    	
		    Solution_vrp solution = new Solution_vrp();
		    solution.distanceMap=distanceMap;
	      	
	        int n_loadboxes = 0;
		    int new_num_nodes = 0;
		    int numberOfCities_ = 0;
		    
		    int[] nodes_record = new int[CLIENT_NUM];
		    
		    for (int i = 0; i < CLIENT_NUM; i++) {
		    	int nk =0;
		    	while (client_v_w[i][0]>1.1*VEHICLE_VOLUME[TRUCKTYPE_NUM-1]) {
		    		client_v_w[i][0] -= truck_weight_ratio[i]*VEHICLE_VOLUME[TRUCKTYPE_NUM-1];
		    		
		    		Route route = new Route((int)(1000+Math.random()*8999));
		    		
		        	ArrayList<Box> unloadboxes = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_trans = new ArrayList<Box>();
					ArrayList<Box> unloadboxes_half_trans = new ArrayList<Box>();
		        	
		        	double volume_precent = 0.0;
		        	int ngoods = clients.get(i).getGoodsNum();
		        	//ArrayList<Box> goods = new ArrayList<Box>();
		        	for (int k = nodes_record[i]; k < ngoods; k++) {
		        		volume_precent += clients.get(i).getGoods().get(k).getVolume();
		        		if( volume_precent >= truck_weight_ratio[i]*VEHICLE_VOLUME[TRUCKTYPE_NUM-1]) {
		        			nodes_record[i] = k;
				        	//nk = k;
		        			break;
		        		}
		        		//goods.add(clients.get(node_id).getGoods().get(k));
		        		unloadboxes.add(new Box(clients.get(i).getGoods().get(k)));
		        		unloadboxes_trans.add(new Box(clients_trans.get(i).getGoods().get(k)));
		        		unloadboxes_half_trans.add(new Box(clients_half_trans.get(i).getGoods().get(k)));
		        		//client_v_w[i][0] -= clients.get(i).getGoods().get(k).getVolume();
		        	}
		        	
        	
		        	LinkedList<Node> nodes = new LinkedList<Node>();
		        	
		        	nodes.add(new Node(depot_start));
		        	nodes.add(new Node(clients.get(i)));
		        	nodes.add(new Node(depot_end));
		        	
		        	route.setNodes(nodes);
		        	
		        	ArrayList<Integer> k = null;
		        	
			        for (int j = 0; j < TRUCKTYPE_NUM; j++) {
						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(j));
						route.setCarriage(vehicle1);	
						boolean if_check = true;
						double unload_v = 0.0;
						double unload_w = 0.0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {
								if_check = false;break;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								if_check = false;break;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								if_check = false;break;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
							unload_w += unloadboxes.get(nbox).getWeight();
						}
						if (unload_v > vehicle1.getTruckVolume()||unload_w>vehicle1.getCapacity()) {
							if_check = false;
						}
						if (if_check) {
							k = route.is_loadable(unloadboxes);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_half_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
						}
			        }
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
		    }//for CLIENT_NUM
		    

		    
			numberOfCities_ = new_num_nodes;
			int[] client_city_map = new int[numberOfCities_];
			     
//			double[][] new_node_coords = new double[numberOfCities_][2];
			double[] new_node_v = new double[numberOfCities_];
			
			
			int var = 2;
			new_num_nodes = 0;
			for (int i = 0; i < CLIENT_NUM; i++) {
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
		        
//		    no_results += 1;
		    	    
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
	        
	        int[] node_if_check = new int[CLIENT_NUM];
	        for (int i = 0; i < CLIENT_NUM; i++) {
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
//		        int nsort = 0;
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
		        for (int j = 0; j < TRUCKTYPE_NUM; j++) {
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
				
		        for (int j = 0; j < n_node[i]; j++) {
		        	
		        	//node上面所有的boxes
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
			        		unloadboxes_node.add(new Box(clients.get(node_id).getGoods().get(k)));
	//		        		Box a = clients.get(node_id).getGoods().get(k);
	//		        		Box b = clients_trans.get(node_id).getGoods().get(k);
			        		unloadboxes_node_trans.add(new Box(clients_trans.get(node_id).getGoods().get(k)));
			        		unloadboxes_node_half_trans.add(new Box(clients_half_trans.get(node_id).getGoods().get(k)));
			        	}
			        	for (int n_unload=0;n_unload<unloadboxes_node.size();n_unload++) {
			        		unloadboxes.add(new Box(unloadboxes_node.get(n_unload)));
			        		unloadboxes_trans.add(new Box(unloadboxes_node_trans.get(n_unload)));
			        		unloadboxes_half_trans.add(new Box(unloadboxes_node_half_trans.get(n_unload)));
			        	}
		        	}
		        }

				nodes.add(new Node(depot_end));				
				route.setNodes(nodes);

				ArrayList<Integer> k = null;
				if (unloadboxes.size() >0) {
			        for (int j = 0; j < TRUCKTYPE_NUM; j++) {
						Carriage vehicle1 = new Carriage(BASIC_TRUCKS.get(j));
						route.setCarriage(vehicle1);	
						boolean if_check = true;
						double unload_v = 0.0;
						double unload_w = 0.0;
						for (int nbox = 0; nbox < unloadboxes.size(); nbox++) {
							if(unloadboxes.get(nbox).getHeight()>vehicle1.getHeight()) {
								if_check = false;break;
							}
							if(unloadboxes.get(nbox).getWidth() >vehicle1.getWidth()) {
								if_check = false;break;
							}
							if(unloadboxes.get(nbox).getLength() >vehicle1.getLength()) {
								if_check = false;break;
							}
							unload_v += unloadboxes.get(nbox).getVolume();
							unload_w += unloadboxes.get(nbox).getWeight();
						}
						if (unload_v > vehicle1.getTruckVolume()||unload_w > vehicle1.getCapacity()) {
							if_check = false;
						}
						if (if_check) {
							k = route.is_loadable(unloadboxes);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
							k = route.is_loadable(unloadboxes_half_trans);
							if (k.size() == unloadboxes.size()) {
								break;
							}
						}
			        }
			                

					if (k!=null) {
						n_loadboxes += k.size();
						solution.getRoutes().add(route);
						if (k.size() !=unloadboxes.size()) {
							break;
						}
					}
				}
	        }//for n_route+1
	        

	        if (n_loadboxes == BOX_NUM) {
	        	solution.evaluation();
	        	solutionSet.add(new Solution_vrp(solution));
	        	save_feasible_no[n_sol] = 1;        	
	        }
	    }
	    return solutionSet;
	}
}// end class