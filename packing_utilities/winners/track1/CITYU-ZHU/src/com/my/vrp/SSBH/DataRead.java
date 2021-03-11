package com.my.vrp.SSBH;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import com.my.vrp.Box;
import com.my.vrp.Carriage;
import com.my.vrp.Node;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class DataRead {
	public static Problem readJSON(String fullFileName){
		JSONParser jsonParser = new JSONParser();
		Problem returnProblem = new Problem();
		try(FileReader reader = new FileReader(fullFileName))
		{
			JSONObject obj = (JSONObject)jsonParser.parse(reader);//最顶层
			Iterator<JSONObject> iterator;//用来遍历JSONObject的Array
			JSONObject algorithmBaseParamDto = (JSONObject)obj.get("algorithmBaseParamDto");
			JSONArray platformDtoList = (JSONArray)algorithmBaseParamDto.get("platformDtoList");

			returnProblem.CLIENT_NUM=-1;
			returnProblem.CLIENT_NUM = platformDtoList.size();
			boolean [] mustFirst = new boolean [returnProblem.CLIENT_NUM];

			//PlatformMap，文件中platform和代码中platformID的对应关系。
			iterator = platformDtoList.iterator();
			int platformID = 0;
			returnProblem.PlatformIDCodeMap.put(platformID, "start_point");
			returnProblem.PlatformCodeIDMap.put("start_point", platformID);
			platformID++;
			while(iterator.hasNext()) {
				JSONObject platform = iterator.next();
				String platformCode = (String)platform.get("platformCode");
				returnProblem.PlatformIDCodeMap.put(platformID, platformCode);
				returnProblem.PlatformCodeIDMap.put(platformCode, platformID);
				mustFirst[platformID-1] = (boolean)platform.get("mustFirst");
				platformID++;
			}
			returnProblem.PlatformIDCodeMap.put(platformID, "end_point");
			returnProblem.PlatformCodeIDMap.put("end_point", platformID);

			returnProblem.TRUCKTYPE_NUM = -1;//汽车的数量
			returnProblem.BASIC_TRUCKS = new ArrayList<>();
			//得到各种类别的卡车
			JSONArray truckTypeDtoList = (JSONArray)algorithmBaseParamDto.get("truckTypeDtoList");
			returnProblem.TRUCKTYPE_NUM = truckTypeDtoList.size();
//			truck_min_max_lwh = new double[3][2];
//			for(int i=0;i<3;i++)
//				truck_min_max_lwh[i][0] = Double.MAX_VALUE;
//			for(int i=0;i<3;i++)
//				truck_min_max_lwh[i][1] = 0;
			for(int basic_truct=0;basic_truct<returnProblem.TRUCKTYPE_NUM;basic_truct++) {
				Carriage truck = new Carriage();
				JSONObject curr_truck = (JSONObject) truckTypeDtoList.get(basic_truct);

				truck.setCapacity((double)curr_truck.get("maxLoad"));
				truck.setHeight((double)curr_truck.get("height"));
				truck.setLength((double)curr_truck.get("length"));
				truck.setWidth((double)curr_truck.get("width"));
				truck.setTruckTypeId((String)curr_truck.get("truckTypeId"));
				truck.setTruckTypeCode((String)curr_truck.get("truckTypeCode"));
				returnProblem.BASIC_TRUCKS.add(truck);
			}

			returnProblem.distanceMap = new HashMap<>();
			//读取distanceMap
			JSONObject distanceMapJSON = (JSONObject)algorithmBaseParamDto.get("distanceMap");
			for(int clienti=1;clienti<=returnProblem.CLIENT_NUM;clienti++) {
				for(int clientj=1;clientj<=returnProblem.CLIENT_NUM;clientj++) {
					if(clienti!=clientj) {
						//不同的client之间的距离从文件里面读取。
						String twoplatforms = returnProblem.PlatformIDCodeMap.get(clienti)+'+'+returnProblem.PlatformIDCodeMap.get(clientj);
//						String twoplatform = clients.get(clienti).getPlatformID());
						returnProblem.distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(clientj), (Double)distanceMapJSON.get(twoplatforms));
					}else {
						//相同的client之间的距离为0.
						returnProblem.distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(clientj), 0.0);
					}
				}
			}
			//从起始点到每个client的距离。
			for(int clienti=1;clienti<=returnProblem.CLIENT_NUM;clienti++) {
				String twoplatforms = returnProblem.PlatformIDCodeMap.get(0)+'+'+returnProblem.PlatformIDCodeMap.get(clienti);
				returnProblem.distanceMap.put(String.valueOf(0)+'+'+String.valueOf(clienti), (Double)distanceMapJSON.get(twoplatforms));
			}
			//从每个client到终点的距离。
			for(int clienti=1;clienti<=returnProblem.CLIENT_NUM;clienti++) {
				String twoplatforms = returnProblem.PlatformIDCodeMap.get(clienti)+'+'+returnProblem.PlatformIDCodeMap.get(returnProblem.CLIENT_NUM+1);
				returnProblem.distanceMap.put(String.valueOf(clienti)+'+'+String.valueOf(returnProblem.CLIENT_NUM+1), (Double)distanceMapJSON.get(twoplatforms));
			}

			returnProblem.depot_start = new Node();//起始节点
			returnProblem.clients = new ArrayList<>();
			returnProblem.depot_end = new Node();//结束节点。
			returnProblem.depot_start.setPlatformID(0);//start no. is always 0
			returnProblem.depot_start.setDemands(0);//start demands is always 0
			returnProblem.depot_start.setGoodsNum(0);//start goodsNum is always 0
			returnProblem.depot_start.setGoods(new ArrayList<Box>());//
			returnProblem.depot_start.setMustFirst(false);

			returnProblem.depot_end.setPlatformID(returnProblem.CLIENT_NUM+1);//
			returnProblem.depot_end.setDemands(0);
			returnProblem.depot_end.setGoodsNum(0);
			returnProblem.depot_end.setGoods(new ArrayList<Box>());
			returnProblem.depot_end.setMustFirst(false);

			//建立clients所有客户，没有箱子需求。
			for(int i=1;i<=returnProblem.CLIENT_NUM;i++) {
				Node client = new Node();
				ArrayList<Box> boxes = new ArrayList<Box>();
				int platform = i;
				client.setPlatformID(platform);//第几个平台，用于distance-matrix的下标。
				client.setDemands(0);//demands==0,the client's demands are boxes
				client.setGoods(boxes);
				client.setGoodsNum(0);//goods num
				client.setLoadgoodsNum(0);//刚开始所有boxes都没有装载。
				returnProblem.clients.add(client);
			}
			//遍历一遍boxes，为客户添加box
			JSONArray boxesJSONArray = (JSONArray)obj.get("boxes");
			iterator = boxesJSONArray.iterator();
			while(iterator.hasNext()) {
				JSONObject currBoxJSON = iterator.next();
				String platformCode = (String)currBoxJSON.get("platformCode");
				//找到它属于哪个客户ID
				platformID = returnProblem.PlatformCodeIDMap.get(platformCode);
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
//				if(width>length) {
//					box.setDirection(200);
////					System.out.println();
////					System.out.println(filenames[fileidx]);
//				}else {
				box.setDirection(100);
//				}
				//为这个客户添加当前的box
				returnProblem.clients.get(platformID-1).getGoods().add(box);
			}

			//最后对每个客户的boxes进行排序，设置GoodsNum
			for(int i=0;i<returnProblem.CLIENT_NUM;i++) {
//				Collections.sort(clients.get(i).getGoods());//按体积进行从大到小进行排序。

				returnProblem.clients.get(i).setGoodsNum(returnProblem.clients.get(i).getGoods().size());//goods num
				returnProblem.clients.get(i).setMustFirst(mustFirst[i]);
			}
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return returnProblem;
	}
	/*
	 * 设置初始值
	 */
	public static Map<String, Double[]> setExtreme(String extremePath) {
		File f = new File(extremePath);
		Map<String, Double[]> idealNadirMap = new HashMap<String, Double[]>();
		if(f.exists()) {
			return readExtreme();
		}else {
			//如果不存在，则都设置初始值。
			f = new File("./data/inputs");
			String[] filenames = f.list();
			for(int fileidx=0;fileidx<filenames.length;fileidx++) {
			Double [] initialPoints = {Double.MAX_VALUE,Double.MAX_VALUE,Double.MIN_VALUE,Double.MIN_VALUE};
			idealNadirMap.put(filenames[fileidx], initialPoints);
			}
			return idealNadirMap;
		}
	}
	/**
	 * 从.\\data\\extremes里面读取ideal and nadir points.
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
		} catch (ParseException e) {
			// TODO Auto-generated catch block 
			e.printStackTrace();
		}
		return idealNadirMap;
	}

}
