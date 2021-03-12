package com.my.vrp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import com.my.vrp.utils.BackSequence;
import com.my.vrp.utils.LeftSequence;
import com.my.vrp.utils.TopSequence;

//import static com.my.vrp.param.Param.*;
/**
 * 车厢类
 * @author dell
 *
 */
public class Carriage  implements Cloneable {

//	private double length = VEHICLE_LENGTH;
//	private double width = VEHICLE_WIDTH;
//	private double height = VEHICLE_HEIGHT;
//	private double capacity = VEHICLE_CAPACITY;
	private double length;
	private double width;
	private double height;
	private double capacity;
	private String truckTypeId;
	private int truckId;
	private String truckTypeCode;
//	private ArrayList<Node> NodeBoxes = new ArrayList<Node>();
	
	
	public Carriage() {
		
	}
	
	public Carriage(Carriage c) {
		this.length=c.length;
		this.width=c.width;
		this.height=c.height;
		this.capacity=c.capacity;
		this.truckTypeId=c.truckTypeId;
		this.truckId = c.truckId;
		this.truckTypeCode=c.truckTypeCode;
//		this.NodeBoxes = new ArrayList<Node>();
//		Iterator<Node> iteratorNode = c.NodeBoxes.iterator();
//		while(iteratorNode.hasNext()) {
//			this.NodeBoxes.add(new Node(iteratorNode.next()));
//		}
		
	}
	
	

	public double getLength() {
		return length;
	}

	public void setLength(double length) {
		this.length = length;
	}
	public double getWidth() {
		return width;
	}

	public void setWidth(double width) {
		this.width = width;
	}

	public double getHeight() {
		return height;
	}

	public void setHeight(double height) {
		this.height = height;
	}

	public double getCapacity() {
		return capacity;
	}

	public void setCapacity(double capacity) {
		this.capacity = capacity;
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		Carriage carriage = null;
		try {
			carriage = (Carriage)super.clone();
		}catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		//这里可以设置不需要的变量。。。
		
		//
		return carriage;
	}


	public double getTruckVolume() {
		return this.width*this.length*this.height;
	}
	public String getTruckTypeCode() {
		return truckTypeCode;
	}

	public void setTruckTypeCode(String truckTypeCode) {
		this.truckTypeCode = truckTypeCode;
	}

	public String getTruckTypeId() {
		return truckTypeId;
	}

	public void setTruckTypeId(String truckTypeId) {
		this.truckTypeId = truckTypeId;
	}

	public int getTruckId() {
		return truckId;
	}

	public void setTruckId(int truckId) {
		this.truckId = truckId;
	}
	
	
/**
 * dblf算法装箱
 * @param clients
 * @return 装箱的box下标数组,装箱个数
 */
//	public ArrayList<Integer> dblfx(ArrayList<Box> boxingSequence) {
//		Random r=new Random();
//		r.setSeed(100);
//		ArrayList<Integer> loadIdx_best = new ArrayList<Integer>();//保存所有迭代中最好的index，有哪些boxes被装载了。
//		ArrayList<Box> thisBoxes_best = new ArrayList<Box>();//保存所有迭代中最好的packing plan
//		boolean[] thisBoxes_best_adjustableFlag=new boolean[boxingSequence.size()];
//		double thisloadWeight_best = 0.0;
//		double thisVI_best = 0.0;//最好解对应的体积覆盖，这个是越大越好。
//		//得到当前进化的box序列。注意不能用clone和addAll，这样会改变boxingSequence
//		ArrayList<Box> evolve_Boxes = new ArrayList<Box>();
//		Iterator<Box> iteratorBox = boxingSequence.iterator();
//		while(iteratorBox.hasNext()) {evolve_Boxes.add(new Box(iteratorBox.next()));}
//		for(int iteration=0;iteration<2;iteration++) {
//			
//		//boxingSequence是请求放在当前小车的箱子序列，每个平台的箱子从大到小排序。
//		ArrayList<Double> horizontal_levels = new ArrayList<Double>();
//		horizontal_levels.add(0.0);
//		ArrayList<Box> sortedBox = new ArrayList<Box>();//清空已经存放的boxes
//		ArrayList<Box> thisBoxes = new ArrayList<Box>();//
//		double thisloadWeight=0.0;
//		ArrayList<Integer> loadIdx=new ArrayList<Integer>();
//		Box zerobox = new Box();
//		zerobox.setHeight(0);
//		zerobox.setLength(0);
//		zerobox.setWidth(0);
//		zerobox.setXCoor(0);
//		zerobox.setYCoor(0);
//		zerobox.setZCoor(0);
//		boolean insertConfirm;//是否成功插入当前箱子。
//		double thisVI = 0.0;//保存检测到的当前plan的覆盖体积。
//		boolean[] adjustableFlag = new boolean[evolve_Boxes.size()];
//		for(int boxi=0;boxi<evolve_Boxes.size();boxi++) {
//			insertConfirm=false;
//			Box curr_box = evolve_Boxes.get(boxi);
//			thisVI = 0.0;//每次添加箱子时都会重新计算。
//			double thisAIK=0.0;
////			double thisAIK_1=0.0;
//			//第一步先求3DCorners=========================================================
//			ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。
//			if(sortedBox.size()<1) {Corners3D.add(new Box(zerobox));}else {
//				
//				ArrayList<Double> Corners2DxLast = new ArrayList<Double>();
//				ArrayList<Double> Corners2DyLast = new ArrayList<Double>();
//				ArrayList<Double> Corners2Dx;
//				ArrayList<Double> Corners2Dy;
//			int k=0;//遍历每个Z平面，和Z轴length垂直的平面。
//			while(k<horizontal_levels.size() && horizontal_levels.get(k)+curr_box.getLength()<=this.length) {
//				thisAIK = 0.0;//保存这个平面下的X-Y覆盖面积。
//				//得到在这个平面之上的已经存放的boxes
//				ArrayList<Box> I_k = new ArrayList<Box>();
//				for(int i=0;i<sortedBox.size();i++) {
//					if(sortedBox.get(i).getZCoor()+sortedBox.get(i).getLength()>horizontal_levels.get(k)) {
//						I_k.add(new Box(sortedBox.get(i)));
//					}
//				}
//				//求2DCorners，在这个过程中，需要计算覆盖的面积AI。=============================================begin
//				Corners2Dx = new ArrayList<Double>();
//				Corners2Dy = new ArrayList<Double>();
//				if(I_k.size()<1) {Corners2Dx.add(0.0);Corners2Dy.add(0.0);}else {
//					//Phase 1: identify the extreme items e_1,...,e_m
//					ArrayList<Integer> e = new ArrayList<Integer>();
//					double bar_x = 0.0;//注意I_k是根据y,x排序的。
//					for(int i=0;i<I_k.size();i++) {
//						if(I_k.get(i).getXCoor()+I_k.get(i).getWidth()>bar_x) {
//							e.add(i);bar_x=I_k.get(i).getXCoor()+I_k.get(i).getWidth();//
//						}
//					}
//					//Phase 2: determine the corner points
//					double XCoor = 0.0;
//					double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//					thisAIK = thisAIK+(I_k.get(e.get(0)).getXCoor()+I_k.get(e.get(0)).getWidth()-XCoor)*YCoor;
//					if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
//						Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//					}
//					for(int j=1;j<e.size();j++) {
//						XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//						YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//						thisAIK = thisAIK+(I_k.get(e.get(j)).getXCoor()+I_k.get(e.get(j)).getWidth()-XCoor)*YCoor;
//						if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
//							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//						}
//					}
//					XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//					YCoor = 0.0;
//					
//					if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
//						Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//					}
//				}
//				//找完了这一层的2DCorners==========================================================end
//				//进行筛选
////				int formerIdx_1=-1;
////				boolean addFlag=false;
//				for(int i=0;i<Corners2Dx.size();i++) {
////					addFlag=false;
//					if(Corners2DxLast.contains(Corners2Dx.get(i))&&Corners2DyLast.contains(Corners2Dy.get(i))) {//如果上一层已经有这个corner了
//						//如果x和y都有的话，看他们的index是否相同。
//						if(Corners2DxLast.indexOf(Corners2Dx.get(i))!=Corners2DyLast.indexOf(Corners2Dy.get(i))){
//							Box corner = new Box(zerobox);
//							corner.setXCoor(Corners2Dx.get(i));
//							corner.setYCoor(Corners2Dy.get(i));
////							addFlag=true;
//							corner.setZCoor(horizontal_levels.get(k));
//							Corners3D.add(corner);
//						}
//					}else {//如果上一层没有这个corner，则增加这个corner。
//						Box corner = new Box(zerobox);
//						corner.setXCoor(Corners2Dx.get(i));
//						corner.setYCoor(Corners2Dy.get(i));
//						//计算面积
////						addFlag=true;
//						corner.setZCoor(horizontal_levels.get(k));
//						Corners3D.add(corner);
//					}
////					if(addFlag) {
//////						formerIdx_2->formerIdx_1->i(current), current may be the last one, so each time, the current one not used.
////						//计算面积。
////						if(formerIdx_1>=0) {
////							thisAIK = thisAIK+(Corners2Dx.get(i)-Corners2Dx.get(formerIdx_1))*Corners2Dy.get(formerIdx_1);
////						}else{
////							thisAIK = Corners2Dx.get(i)*this.height;
////						}
////						formerIdx_1=i;
////					}
//				}//筛选完毕
////				if(formerIdx_1>=0) {
////				thisAIK=thisAIK+(this.width-Corners2Dx.get(formerIdx_1)*Corners2Dy.get(formerIdx_1));
////				}
//				//save this Corners2D
//				Corners2DxLast = (ArrayList<Double>)Corners2Dx.clone();
//				Corners2DyLast = (ArrayList<Double>)Corners2Dy.clone();
//				//==========================计算覆盖的体积。(z'_k-z'_{k-1})*AI_{k-1}
////				if(k>=1) {
////					thisVI=thisVI+(horizontal_levels.get(k)-horizontal_levels.get(k-1))*thisAIK_1;
////				}
////				thisAIK_1=thisAIK;
//				if(k+1<horizontal_levels.size())
//					thisVI=thisVI+(horizontal_levels.get(k+1)-horizontal_levels.get(k))*thisAIK;
//				//==========================
//				k=k+1;
//			}//for k
////			thisVI=thisVI+(this.length-horizontal_levels.get(k-1))*thisAIK;
//			}
//			//第一步先求3DCorners=========================================================end
//			//找一个位置去存放当前箱子boxingSeqence.get(boxi)。
//			//将这些位置随机打乱。
////			int [] permutation= com.my.vrp.utils.Permutation.intPermutation(Corners3D.size());
//			for(int positioni=0;positioni<Corners3D.size();positioni++) {
//				Box curr_position = Corners3D.get(positioni);
//				if(curr_position.getXCoor()+curr_box.getWidth()<=this.width&&
//						curr_position.getYCoor()+curr_box.getHeight()<=this.height&&
//								curr_position.getZCoor()+curr_box.getLength()<=this.length) {
//					//判断这个位置能不能站稳
//					//当前箱子的坐标： boxingSequence.x,y,z
//					//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//					//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//					boolean support = false;
//					if(curr_position.getYCoor()==0) {
//						support = true;
//					}else{
//						//计算该箱子的底部面积。
////						Box currBox = boxingSequence.get(i);
//						double bottomArea = curr_box.getWidth()*curr_box.getLength();
//						double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
//						double crossArea = 0;
//						//计算所有已放箱子的顶部与该箱子的底部交叉面积
//						
//						for (int boxii=0;boxii<sortedBox.size();boxii++) {
//							//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//							Box existBox = sortedBox.get(boxii);
//							
//							if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=0.001) {
//								double xc=curr_position.getXCoor(),yc=curr_position.getYCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ye=existBox.getYCoor(),ze=existBox.getZCoor();
//								double wc=curr_box.getWidth(),hc=curr_box.getHeight(),lc=curr_box.getLength(),we=existBox.getWidth(),he=existBox.getHeight(),le=existBox.getLength();
//								
//								if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
//									double [] XCoor = {xc,xc+wc,xe,xe+we};
//									double [] ZCoor = {zc,zc+lc,ze,ze+le};
//									//sort xc,xc+wc,xe,xe+we
//									 Arrays.sort(XCoor);
//									 Arrays.sort(ZCoor);
//									//sort zc,zc+lc,ze,ze+le
//									 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//									 if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%并且已经有交叉了，则不用继续判断了。
//								}
////								if((xc+wc>xe)&&(ze+le>zc)) {
////									crossArea = crossArea+Math.min(xc+wc-xe,wc)*Math.min(ze+le-zc,lc);
////								}
////								if((xe+we>xc)&&(zc+lc>ze)) {
////									crossArea = crossArea+Math.min(xe+we-xc,wc)*Math.min(zc+lc-ze,lc);
////								}
//							}
//							
//						}
//						
//					}
//					boolean flagXYcross = false;
//					if(curr_position.getZCoor()>curr_box.getLength()) {
//						for (int boxii=0;boxii<sortedBox.size();boxii++) {
//							//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//							Box existBox = sortedBox.get(boxii);
//							if(Math.abs(existBox.getZCoor()+existBox.getLength()-curr_position.getZCoor())<=0.1) {
//								double xc=curr_position.getXCoor(),yc=curr_position.getYCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ye=existBox.getYCoor(),ze=existBox.getZCoor();
//								double wc=curr_box.getWidth(),hc=curr_box.getHeight(),lc=curr_box.getLength(),we=existBox.getWidth(),he=existBox.getHeight(),le=existBox.getLength();
//								//判断当前箱子和已经存在的箱子在X-Y平面是否有交叉。如果已经有交叉，则不用继续判断了。如果没有交叉则需要继续判断，直到最后。
//								if(!((xc+wc<xe)||(xe+we<xc)||(yc+hc<ye)||(ye+he<yc))) {
//									flagXYcross = true;
//								}
//							}
//						}
//					}
//					//
//					if(support) {
//						Box loadBox = new Box(curr_box);
//						loadBox.setXCoor(curr_position.getXCoor());
//						loadBox.setYCoor(curr_position.getYCoor());
//						loadBox.setZCoor(curr_position.getZCoor());
//						int idx=0;
//						for(idx=0;idx<sortedBox.size();idx++) {//按y,x,z来排序。
//							Box thisbox = sortedBox.get(idx);
//							//如果在一个水平面上，则对比X
//							if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<10) {
//								if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
//							}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//								break;
//							}
////							if(&&Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())>10) {
////								//对Y进行排序
////								break;
////							}else if(&&) {//对X进行排序。
////								break;
////							}//else if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<0.01&&
//							//		Math.abs(thisbox.getXCoor()+thisbox.getWidth()-loadBox.getXCoor()-loadBox.getWidth())<0.01&&
//							//		thisbox.getZCoor()+this.getLength()>loadBox.getZCoor()+loadBox.getLength()) {
//							//	break;
//							//}
//						}
//						if(idx==sortedBox.size()) {
//							sortedBox.add(new Box(loadBox));//sort sortedBox by y,x coordinate
//						}else {
//							sortedBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//						}
//						double curr_level = loadBox.getZCoor()+loadBox.getLength();
//						if(!horizontal_levels.contains(curr_level)) {
//							
//							for(idx=0;idx<horizontal_levels.size();idx++) {
//								if(horizontal_levels.get(idx)>curr_level) {
//									break;
//								}
//							}
//							if(idx==horizontal_levels.size()) {
//								horizontal_levels.add(curr_level);
//							}else {
//								horizontal_levels.add(idx, curr_level);
//							}
//						}
//						//if this horizontal level loadBox.y+loadBox.height not in horizontal_levels, then add it and sort horizontal levels by increasing values
//						double weights = thisloadWeight + loadBox.getWeight();
//						if(weights>this.capacity) {
////							return loadIdx;
//							insertConfirm=false;
//							break;//推出寻找最优位置，并且退出遍历箱子。
//						}
//						thisBoxes.add(loadBox);
//						thisloadWeight = thisloadWeight + loadBox.getWeight();
//						loadIdx.add(boxi);//第i-2+1个箱子被装载了。
//						insertConfirm=true;
//						if(!flagXYcross)//如果这个箱子的前面（靠近驾驶座这边）没有箱子，则这个箱子的位置可以调前面，在下次迭代时看是否能够放到前面。
//							adjustableFlag[boxi]=true;
//						break;
//					}
//				}
//			}
//			if(!insertConfirm) {break;}
//		}//for boxi
//		//如果这次的plan is better, than replace.
//		if(thisVI>thisVI_best) {
//			thisBoxes_best = thisBoxes;
//			thisBoxes_best_adjustableFlag = adjustableFlag;
//			thisloadWeight_best=thisloadWeight;
//			thisVI_best = thisVI;
//			loadIdx_best = loadIdx;
//		}
//		//下一次迭代之前，先进行调换顺序。
//		//将thisBoxes_best复制到evolve_Boxes，然后再根据thisBoxes_best_adjustableFlag进行调换。
//		evolve_Boxes = new ArrayList<Box>();
////		evolve_Boxes = thisBoxes_best.clone();//千万注意不能这样用，因为等下evolve_Boxes是需要进行调换顺序的。
////		iteratorBox = thisBoxes_best.iterator();
////		while(iteratorBox.hasNext()) {evolve_Boxes.add(new Box(iteratorBox.next()));}
//		for(int boxi=0;boxi<thisBoxes_best.size();boxi++) {
//			if(thisBoxes_best_adjustableFlag[boxi]) {//如果前面检测到这个box的位置可以调整。那么这个box前面Z+height比z+height高的箱子前面。
//				//依概率进行调整
//				int boxii=boxi-1;
//				if(r.nextDouble()<=1.0) {
//					//往前遍历evolve_Boxes
//					while(boxii>0&&evolve_Boxes.get(boxii).getPlatformid()==thisBoxes_best.get(boxi).getPlatformid())
////					for(int ;;boxii--) {
//					if(evolve_Boxes.get(boxii).getYCoor()+evolve_Boxes.get(boxii).getHeight()>thisBoxes_best.get(boxi).getYCoor()+thisBoxes_best.get(boxi).getHeight()+10) {
//						//进行调整。
//						boxii=boxii-1;
//					}else {
//						break;
//					}
////					}
//				}
//				if(boxii==boxi-1) {
//					evolve_Boxes.add(new Box(thisBoxes_best.get(boxi)));//在最后添加。
//				}else {
//					evolve_Boxes.add(boxii, new Box(thisBoxes_best.get(boxi)));//在该位置添加，并将该位置及其后面的items往后移动。
//				}
//				
//			}else {
//				evolve_Boxes.add(new Box(thisBoxes_best.get(boxi)));
//			}
//		}
//		}//iteration
//		this.Boxes = thisBoxes_best;
//		this.loadWeight=thisloadWeight_best;
//		//calculate excessWeight
//		if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//		//calculate excessLength.
////		if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
////				+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
////			this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
////					+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
////		else
//			this.excessLength = 0;
//		
//		
////		System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////		if(left.leftSequence.size()<boxingSequence1.size())
////		System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//		return loadIdx_best;//left.leftSequence.size();
//	}
	
	
	
	
	
//	/**
//	 * dblf算法装箱,根据yx进行排序找3DCorners
//	 * @param clients
//	 * @return 装箱的box下标数组,装箱个数
//	 */
//		public ArrayList<Integer> dblf(ArrayList<Box> boxingSequence) {
//			ArrayList<Box> thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
//			ArrayList<Box> thissortedBox = new ArrayList<Box>();
//			ArrayList<Double> horizontal_levels = new ArrayList<Double>();
//			ArrayList<Integer> loadIdx=new ArrayList<Integer>();//保存装在这辆车里面的箱子集
//			double thisloadWeight=0.0;//保存已经装的箱子的重量
//			int iter=0;
//			while(iter<1) {
//			//boxingSequence是请求放在当前小车的箱子序列，每个平台的箱子从大到小排序。
//			horizontal_levels = new ArrayList<Double>();
//			horizontal_levels.add(0.0);
//			thissortedBox = new ArrayList<Box>();//清空已经存放的boxes
//			thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
//			thisloadWeight=0.0;//保存已经装的箱子的重量
//			loadIdx=new ArrayList<Integer>();//保存装在这辆车里面的箱子集
//			Iterator<Box> iteratorBox;
//			boolean insertConfirm;//是否成功插入当前箱子。
//			for(int boxi=0;boxi<boxingSequence.size();boxi++) {
//				insertConfirm=false;
//				Box curr_box = boxingSequence.get(boxi);
//				if(thisloadWeight + curr_box.getWeight()>this.capacity) {
//					break;//当前箱子不能再加入这辆车了，退出寻找最优位置，并且退出遍历箱子。
//				}
//				//第一步先求3DCorners=========================================================
//				ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。
////				if(sortedBox.size()<1) {
////					Corners3D.add(new Box());
////				} else {
////				int k=0;//遍历每个Z平面，和Z轴length垂直的平面。
//				for(int k=0;k<horizontal_levels.size() && horizontal_levels.get(k)+curr_box.getLength()<=this.length;k++) {
//					//得到在这个平面之上的已经存放的boxes,I_k
//					ArrayList<Box> I_k = new ArrayList<Box>();
//					iteratorBox = thissortedBox.iterator();
//					while(iteratorBox.hasNext()) {
//						Box currBox = iteratorBox.next();
//						if(currBox.getZCoor()+currBox.getLength()>horizontal_levels.get(k)) {
//							I_k.add(new Box(currBox));
//						}
//					}
//					
//					//求2DCorners==========================================================begin
//					if(I_k.size()<1) {
//						//如果这个平面之上没有box,添加原点。
//						Box corner = new Box();
//						corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//						Corners3D.add(corner);
//					}else{
//						//Phase 1: identify the extreme items e_1,...,e_m
//						ArrayList<Integer> e = new ArrayList<Integer>();
//						double bar_x = 0.0;//注意I_k是根据y,x排序的。
//						for(int i=0;i<I_k.size();i++) {
//							if(I_k.get(i).getXCoor()+I_k.get(i).getWidth()>bar_x) {
//								e.add(i);bar_x=I_k.get(i).getXCoor()+I_k.get(i).getWidth();//
//							}
//						}
//						//Phase 2: determine the corner points
//						double XCoor = 0.0;
//						double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//						if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//							Box corner = new Box();
//							corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//							Corners3D.add(corner);
//						}
//						for(int j=1;j<e.size();j++) {
//							XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//							YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//								Box corner = new Box();
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//								Corners3D.add(corner);
//							}
//						}
//						XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//						YCoor = 0.0;
//						if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//							Box corner = new Box();
//							corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//							Corners3D.add(corner);
//						}
//					}
//				}//for k (each horizontal level)
//				
//				
//				
////				}// if I
//				//结束求解3DCorners=========================================================end
//				//找第一个能够存放当前箱子的位置存放当前箱子。
//				iteratorBox = Corners3D.iterator();
//				while(iteratorBox.hasNext()) {
//					Box curr_position = iteratorBox.next();
//					if(curr_position.getXCoor()+curr_box.getWidth()<=this.width&&
//							curr_position.getYCoor()+curr_box.getHeight()<=this.height&&
//									curr_position.getZCoor()+curr_box.getLength()<=this.length) {
//						//判断这个位置能不能站稳
//						//当前箱子的坐标： boxingSequence.x,y,z
//						//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//						//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//						boolean support = false;
//						if(curr_position.getYCoor()==0) {
//							support = true;
//						}else{
//							//计算该箱子的底部面积。
//							double bottomArea = curr_box.getWidth()*curr_box.getLength();
//							double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
//							double crossArea = 0;
//							//计算所有已放箱子的顶部与该箱子的底部交叉面积
//							for (int boxii=0;boxii<thissortedBox.size();boxii++) {
//								//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//								Box existBox = thissortedBox.get(boxii);
//								
//								if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=1.5) {
//									double xc=curr_position.getXCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ze=existBox.getZCoor();
//									double wc=curr_box.getWidth(),lc=curr_box.getLength(),we=existBox.getWidth(),le=existBox.getLength();
//									
//									if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
//										double [] XCoor = {xc,xc+wc,xe,xe+we};
//										double [] ZCoor = {zc,zc+lc,ze,ze+le};
//										//sort xc,xc+wc,xe,xe+we
//										 Arrays.sort(XCoor);
//										 Arrays.sort(ZCoor);
//										//sort zc,zc+lc,ze,ze+le
//										 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//										 if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%，则不用继续判断了。
//									}
//								}
//							}
//							
//						}
//						//
//						if(support) {//当前箱子可以加入到这辆车中。
//							Box loadBox = new Box(curr_box);
//							loadBox.setXCoor(curr_position.getXCoor());
//							loadBox.setYCoor(curr_position.getYCoor());
//							loadBox.setZCoor(curr_position.getZCoor());
//							
//							//将这个箱子插入到sortedBox里面，按Y-X从大到小进行排序。
//							int idx=0;
//							for(idx=0;idx<thissortedBox.size();idx++) {//按y,x,z来排序。
//								Box thisbox = thissortedBox.get(idx);
//								//如果在一个水平面上，则对比X
//								if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<1.5) {
//									if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
//								}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//									break;
//								}
//							}
//							thissortedBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//							//新增水平面。
//							double curr_level = loadBox.getZCoor()+loadBox.getLength();
//							boolean addFlag=true;
//							for(idx=curr_position.getPlatformid();idx<horizontal_levels.size();idx++) {
//								if(Math.abs(horizontal_levels.get(idx)-curr_level)<50) {//两个level相差多远就不加入来了。
//									addFlag=false;break;
//								}else if(horizontal_levels.get(idx)>curr_level) {
//									break;
//								}
//							}
//							if(addFlag) horizontal_levels.add(idx, curr_level);
////							if(!horizontal_levels.contains(curr_level)) {
////								
////								for(idx=curr_position.getPlatformid();idx<horizontal_levels.size();idx++) {
////									if(horizontal_levels.get(idx)>curr_level) {
////										break;
////									}
////								}
////								horizontal_levels.add(idx, curr_level);
////							}
//							//if this horizontal level loadBox.y+loadBox.height not in horizontal_levels, then add it and sort horizontal levels by increasing values
//							
//							thisBoxes.add(loadBox);
//							thisloadWeight = thisloadWeight + loadBox.getWeight();
//							loadIdx.add(boxi);//第i-2+1个箱子被装载了。
//							insertConfirm=true;
//							break;
//						}
//					}
//				}
//				if(!insertConfirm) {break;}
//			}
//			if(loadIdx.size()==boxingSequence.size()) break;
//			else {
//				//调整序列的顺序，和方向。
//				
//			}
//			iter++;
//			}//while true
//			this.Boxes = thisBoxes;
//			this.sortedBoxes = thissortedBox;
//			this.horizontal_levels = horizontal_levels;
//			this.loadWeight=thisloadWeight;
//			//calculate excessWeight
//			if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//			//calculate excessLength.
////			if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
////					+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
////				this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
////						+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
////			else
//				this.excessLength = 0;
////			System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////			if(left.leftSequence.size()<boxingSequence1.size())
////			System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//			return loadIdx;//left.leftSequence.size();
//		}
//	
//	
//		/**
//		 * 添加boxingSequence中的boxes到this.Boxes
//		 * dblf算法装箱,根据yx进行排序找3DCorners
//		 * @param clients
//		 * @return 装箱的box下标数组,装箱个数
//		 */
//			public ArrayList<Integer> dblfAdd(ArrayList<Box> boxingSequence) {
//				Iterator<Box> iteratorBox;
//				ArrayList<Box> thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
//				ArrayList<Box> thissortedBox = new ArrayList<Box>();//
//				ArrayList<Double> horizontal_levels = new ArrayList<Double>();//
//				ArrayList<Integer> loadIdx=new ArrayList<Integer>();//保存装在这辆车里面的箱子集
//				double thisloadWeight=0.0;//保存已经装的箱子的重量
//				int iter=0;
//				while(iter<1) {
//				//boxingSequence是请求放在当前小车的箱子序列，每个平台的箱子从大到小排序。
//				horizontal_levels = new ArrayList<Double>();
//				Iterator<Double> iteratorDouble = this.horizontal_levels.iterator();//new ArrayList<Double>();
//				while(iteratorDouble.hasNext()) {horizontal_levels.add(iteratorDouble.next());}
////				horizontal_levels.add(0.0);
//				
//				thissortedBox = new ArrayList<Box>();//清空已经存放的boxes
//				thisBoxes = new ArrayList<Box>();//按顺序保存该箱子。
//				
//				//加载sortedBox
//				iteratorBox = this.sortedBoxes.iterator();
//				while(iteratorBox.hasNext()) {
//					thissortedBox.add(new Box(iteratorBox.next()));
//				}
//				//加载this.Boxes
//				iteratorBox = this.Boxes.iterator();
//				while(iteratorBox.hasNext()) {
//					thisBoxes.add(new Box(iteratorBox.next()));
//				}
//				
//				thisloadWeight=this.loadWeight;//保存已经装的箱子的重量
//				loadIdx=new ArrayList<Integer>();//保存装在这辆车里面的箱子集
//				
//				boolean insertConfirm;//是否成功插入当前箱子。
//				for(int boxi=0;boxi<boxingSequence.size();boxi++) {
//					insertConfirm=false;
//					Box curr_box = boxingSequence.get(boxi);
//					if(thisloadWeight + curr_box.getWeight()>this.capacity) {
//						break;//当前箱子不能再加入这辆车了，退出寻找最优位置，并且退出遍历箱子。
//					}
//					//第一步先求3DCorners=========================================================
//					ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。
////					if(sortedBox.size()<1) {
////						Corners3D.add(new Box());
////					} else {
////					int k=0;//遍历每个Z平面，和Z轴length垂直的平面。
//					for(int k=0;k<horizontal_levels.size() && horizontal_levels.get(k)+curr_box.getLength()<=this.length;k++) {
//						//得到在这个平面之上的已经存放的boxes,I_k
//						ArrayList<Box> I_k = new ArrayList<Box>();
//						iteratorBox = thissortedBox.iterator();
//						while(iteratorBox.hasNext()) {
//							Box currBox = iteratorBox.next();
//							if(currBox.getZCoor()+currBox.getLength()>horizontal_levels.get(k)) {
//								I_k.add(new Box(currBox));
//							}
//						}
//						
//						//求2DCorners==========================================================begin
//						if(I_k.size()<1) {
//							//如果这个平面之上没有box,添加原点。
//							Box corner = new Box();
//							corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//							Corners3D.add(corner);
//						}else{
//							//Phase 1: identify the extreme items e_1,...,e_m
//							ArrayList<Integer> e = new ArrayList<Integer>();
//							double bar_x = 0.0;//注意I_k是根据y,x排序的。
//							for(int i=0;i<I_k.size();i++) {
//								if(I_k.get(i).getXCoor()+I_k.get(i).getWidth()>bar_x) {
//									e.add(i);bar_x=I_k.get(i).getXCoor()+I_k.get(i).getWidth();//
//								}
//							}
//							//Phase 2: determine the corner points
//							double XCoor = 0.0;
//							double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//								Box corner = new Box();
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//								Corners3D.add(corner);
//							}
//							for(int j=1;j<e.size();j++) {
//								XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//								YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//								if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									Box corner = new Box();
//									corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//									Corners3D.add(corner);
//								}
//							}
//							XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//							YCoor = 0.0;
//							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//								Box corner = new Box();
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));corner.setPlatformid(k);//记录哪个level
//								Corners3D.add(corner);
//							}
//						}
//					}//for k (each horizontal level)
//					
//					
//					
////					}// if I
//					//结束求解3DCorners=========================================================end
//					//找第一个能够存放当前箱子的位置存放当前箱子。
//					iteratorBox = Corners3D.iterator();
//					while(iteratorBox.hasNext()) {
//						Box curr_position = iteratorBox.next();
//						if(curr_position.getXCoor()+curr_box.getWidth()<=this.width&&
//								curr_position.getYCoor()+curr_box.getHeight()<=this.height&&
//										curr_position.getZCoor()+curr_box.getLength()<=this.length) {
//							//判断这个位置能不能站稳
//							//当前箱子的坐标： boxingSequence.x,y,z
//							//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//							//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//							boolean support = false;
//							if(curr_position.getYCoor()==0) {
//								support = true;
//							}else{
//								//计算该箱子的底部面积。
//								double bottomArea = curr_box.getWidth()*curr_box.getLength();
//								double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
//								double crossArea = 0;
//								//计算所有已放箱子的顶部与该箱子的底部交叉面积
//								for (int boxii=0;boxii<thissortedBox.size();boxii++) {
//									//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//									Box existBox = thissortedBox.get(boxii);
//									
//									if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=1.5) {
//										double xc=curr_position.getXCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ze=existBox.getZCoor();
//										double wc=curr_box.getWidth(),lc=curr_box.getLength(),we=existBox.getWidth(),le=existBox.getLength();
//										
//										if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
//											double [] XCoor = {xc,xc+wc,xe,xe+we};
//											double [] ZCoor = {zc,zc+lc,ze,ze+le};
//											//sort xc,xc+wc,xe,xe+we
//											 Arrays.sort(XCoor);
//											 Arrays.sort(ZCoor);
//											//sort zc,zc+lc,ze,ze+le
//											 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//											 if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%，则不用继续判断了。
//										}
//									}
//								}
//								
//							}
//							//
//							if(support) {//当前箱子可以加入到这辆车中。
//								Box loadBox = new Box(curr_box);
//								loadBox.setXCoor(curr_position.getXCoor());
//								loadBox.setYCoor(curr_position.getYCoor());
//								loadBox.setZCoor(curr_position.getZCoor());
//								
//								//将这个箱子插入到sortedBox里面，按Y-X从大到小进行排序。
//								int idx=0;
//								for(idx=0;idx<thissortedBox.size();idx++) {//按y,x,z来排序。
//									Box thisbox = thissortedBox.get(idx);
//									//如果在一个水平面上，则对比X
//									if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<1.5) {
//										if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
//									}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//										break;
//									}
//								}
//								thissortedBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//								//新增水平面。
//								double curr_level = loadBox.getZCoor()+loadBox.getLength();
//								boolean addFlag=true;
//								for(idx=curr_position.getPlatformid();idx<horizontal_levels.size();idx++) {
//									if(Math.abs(horizontal_levels.get(idx)-curr_level)<50) {//两个level相差多远就不加入来了。
//										addFlag=false;break;
//									}else if(horizontal_levels.get(idx)>curr_level) {
//										break;
//									}
//								}
//								if(addFlag) horizontal_levels.add(idx, curr_level);
////								if(!horizontal_levels.contains(curr_level)) {
////									
////									for(idx=curr_position.getPlatformid();idx<horizontal_levels.size();idx++) {
////										if(horizontal_levels.get(idx)>curr_level) {
////											break;
////										}
////									}
////									horizontal_levels.add(idx, curr_level);
////								}
//								//if this horizontal level loadBox.y+loadBox.height not in horizontal_levels, then add it and sort horizontal levels by increasing values
//								
//								thisBoxes.add(loadBox);
//								thisloadWeight = thisloadWeight + loadBox.getWeight();
//								loadIdx.add(boxi);//第i-2+1个箱子被装载了。
//								insertConfirm=true;
//								break;
//							}
//						}
//					}
//					if(!insertConfirm) {break;}
//				}
//				if(loadIdx.size()==boxingSequence.size()) break;
//				else {
//					//调整序列的顺序，和方向。
//					
//				}
//				iter++;
//				}//while true
//				this.Boxes = thisBoxes;
//				this.sortedBoxes = thissortedBox;
//				this.horizontal_levels = horizontal_levels;
//				this.loadWeight=thisloadWeight;
//				//calculate excessWeight
//				if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//				//calculate excessLength.
////				if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
////						+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
////					this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
////							+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
////				else
//					this.excessLength = 0;
////				System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////				if(left.leftSequence.size()<boxingSequence1.size())
////				System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//				return loadIdx;//left.leftSequence.size();
//			}
		
	
	
		
		
		
		/**
		 * dblf算法装箱,根据yx，yz进行排序找3DCorners
		 * @param clients
		 * @return 装箱的box下标数组,装箱个数
		 */
//			public ArrayList<Integer> dblfyxyz(ArrayList<Box> boxingSequence) {
//				//boxingSequence是请求放在当前小车的箱子序列，每个平台的箱子从大到小排序。
//				ArrayList<Double> yx_levels = new ArrayList<Double>();
//				yx_levels.add(0.0);
//				ArrayList<Double> yz_levels = new ArrayList<Double>();
//				yz_levels.add(0.0);
//				ArrayList<Box> sortedYXBox = new ArrayList<Box>();//清空已经存放的boxes
//				ArrayList<Box> sortedYZBox = new ArrayList<Box>();//清空已经存放的boxes
//				ArrayList<Box> thisBoxes = new ArrayList<Box>();//
//				double thisloadWeight=0.0;
//				ArrayList<Integer> loadIdx=new ArrayList<Integer>();
//				Box zerobox = new Box();
//				zerobox.setHeight(0);
//				zerobox.setLength(0);
//				zerobox.setWidth(0);
//				zerobox.setXCoor(0);
//				zerobox.setYCoor(0);
//				zerobox.setZCoor(0);
//				zerobox.setDirection(100);
//				boolean insertConfirm;//是否成功插入当前箱子。
//				for(int boxi=0;boxi<boxingSequence.size();boxi++) {
//					insertConfirm=false;
//					Box curr_box = boxingSequence.get(boxi);
//					if(thisloadWeight + curr_box.getWeight()>this.capacity) {
//						break;//推出寻找最优位置，并且退出遍历箱子。
//					}
//					//第一步先求3DCorners=========================================================
//					ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。
//					ArrayList<Double> sumWHL = new ArrayList<Double>();
//					
//					
//					//对YX平面。
//					if(sortedYXBox.size()<1) {
//						Corners3D.add(new Box(zerobox));sumWHL.add(0.0);
//					}
////					else if (sortedBox.size()<2){
////						if(sortedBox.get(0).getHeight()+curr_box.getHeight()<=this.height) {
//////							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
////							Box corner = new Box(zerobox);
////							corner.setXCoor(0.0);corner.setYCoor(sortedBox.get(0).getHeight());corner.setZCoor(0.0);Corners3D.add(corner);
////						}
////						if(sortedBox.get(0).getWidth()+curr_box.getWidth()<=this.width) {
////						Box corner = new Box(zerobox);
////						corner.setXCoor(sortedBox.get(0).getWidth());corner.setYCoor(0.0);corner.setZCoor(0.0);Corners3D.add(corner);
////						}
////						Box corner = new Box(zerobox);
////						corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(sortedBox.get(0).getLength());Corners3D.add(corner);
//////						for(int j=1;j<e.size();j++) {
//////							XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//////							YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//////							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//////								Box corner = new Box(zerobox);
//////								corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));Corners3D.add(corner);
//////							}
//////						}
//////						XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//////						YCoor = 0.0;
//////						if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////////							Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//////							Box corner = new Box(zerobox);
//////							corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));Corners3D.add(corner);
//////						}
//////						Corners3D.add();
////					}
//					else {
//						
////						ArrayList<Double> Corners2DxLast = new ArrayList<Double>();
////						ArrayList<Double> Corners2DyLast = new ArrayList<Double>();
////						ArrayList<Double> Corners2Dx;
////						ArrayList<Double> Corners2Dy;
////					int k=0;//遍历每个Z平面，和Z轴length垂直的平面。
//					//保存每一层所有的箱子。
//					ArrayList<ArrayList<Box>> I_K=new ArrayList<ArrayList<Box>>();//保存倒数第二个平面上的箱子。用来处理最后一个。
//					for(int k=0;k<yx_levels.size() && yx_levels.get(k)+curr_box.getLength()<=this.length;k++) {
//						//得到在这个平面之上的已经存放的boxes
//						ArrayList<Box> I_k = new ArrayList<Box>();
//						for(int i=0;i<sortedYXBox.size();i++) {
//							if(sortedYXBox.get(i).getZCoor()+sortedYXBox.get(i).getLength()>yx_levels.get(k)) {
//								I_k.add(new Box(sortedYXBox.get(i)));
//							}
//						}
////						if(k==yx_levels.size()-2) {
////							
////						}
//						//求2DCorners==========================================================begin
////						Corners2Dx = new ArrayList<Double>();
////						Corners2Dy = new ArrayList<Double>();
//						if(I_k.size()<1) {
////							Corners2Dx.add(0.0);Corners2Dy.add(0.0);
////							if(k>=1&&k==yx_levels.size()-1) {//如果是最后一层，检查是否可以向前推进。
////								//x=curr_box.getWidth()<all lastbutoneBoxes.x //yx_levels.get(k)
////								int formeri=0;
////								for(formeri=0;formeri<lastbutoneBoxes.size();formeri++) {
////									if(!(curr_box.getWidth()<=lastbutoneBoxes.get(formeri).getXCoor())) {
////										break;//找到了这种的话，提前结束。
////									}
////								}
////								double thislevel = yx_levels.get(k);
////								if(formeri==lastbutoneBoxes.size()) {
////									//不存在会和这个box交叉的箱子。
////									//将这个点往前移动一个level
////									thislevel = yx_levels.get(k-1);
////								}
////								Box corner = new Box(zerobox);
////								corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(thislevel);
////								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////							}else {
//							double XCoor = 0.0;
//							double YCoor = 0.0;
//							Box corner = new Box(zerobox);
//							corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//							//find a level for ZCoor, k,k-1,k-2
//							int currLevel = k;
////							boolean levelFound=false;
//							for(int currk=k-1;currk>=0;currk--) {
//								ArrayList<Box> I = I_K.get(currk);
//								int i=0;
//								for(i=0;i<I.size();i++) {
//									if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//										//this box should parallel with new box.
//										if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//											break;//找到了这种的话，提前结束。
//										}
//									}
//								}
//								if(i==I.size()) {currLevel = currk;}else{break;}
//							}
//							corner.setZCoor(yx_levels.get(currLevel));
//							Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////							}
//						}else {
//							//Phase 1: identify the extreme items e_1,...,e_m
//							ArrayList<Integer> e = new ArrayList<Integer>();
//							double bar_x = 0.0;//注意I_k是根据y,x排序的。
//							for(int i=0;i<I_k.size();i++) {
//								if(I_k.get(i).getXCoor()+I_k.get(i).getWidth()>bar_x) {
//									e.add(i);bar_x=I_k.get(i).getXCoor()+I_k.get(i).getWidth();//
//								}
//							}
//							//Phase 2: determine the corner points
//							double XCoor = 0.0;
//							double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//								Box corner = new Box(zerobox);
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//								//find a level for ZCoor, k,k-1,k-2
//								int currLevel = k;
////								boolean levelFound=false;
//								for(int currk=k-1;currk>=0;currk--) {
//									ArrayList<Box> I = I_K.get(currk);
//									int i=0;
//									for(i=0;i<I.size();i++) {
//										if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//											//this box should parallel with new box.
//											if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//												break;//找到了这种的话，提前结束。
//											}
//										}
//									}
//									if(i==I.size()) {currLevel = currk;}else{break;}
//								}
//								corner.setZCoor(yx_levels.get(currLevel));
//								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//							}
//							for(int j=1;j<e.size();j++) {
//								XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//								YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//								if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									Box corner = new Box(zerobox);
//									corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//									//find a level for ZCoor, k,k-1,k-2
//									int currLevel = k;
////									boolean levelFound=false;
//									for(int currk=k-1;currk>=0;currk--) {
//										ArrayList<Box> I = I_K.get(currk);
//										int i=0;
//										for(i=0;i<I.size();i++) {
//											if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//												//this box should parallel with new box.
//												if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//													break;//找到了这种的话，提前结束。
//												}
//											}
//										}
//										if(i==I.size()) {currLevel = currk;}else{break;}
//									}
//									corner.setZCoor(yx_levels.get(currLevel));
//									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//								}
//							}
//							XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//							YCoor = 0.0;
//							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//								Box corner = new Box(zerobox);
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//								//find a level for ZCoor, k,k-1,k-2
//								int currLevel = k;
////								boolean levelFound=false;
//								for(int currk=k-1;currk>=0;currk--) {
//									ArrayList<Box> I = I_K.get(currk);
//									int i=0;
//									for(i=0;i<I.size();i++) {
//										if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//											//this box should parallel with new box.
//											if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//												break;//找到了这种的话，提前结束。
//											}
//										}
//									}
//									if(i==I.size()) {currLevel = currk;}else{break;}
//								}
//								corner.setZCoor(yx_levels.get(currLevel));
//								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//							}
//						}
//						//保存这一层的箱子。
//						I_K.add(I_k);
//					}//k
//					//重新调整所有的Corners3D,看yx-level是否可以往前移动。
//					
//					//找yz这一层的。
////					for(int k=0;k<yz_levels.size() && yz_levels.get(k)+curr_box.getWidth()<=this.width;k++) {
////						//得到在这个平面之上的已经存放的boxes
////						ArrayList<Box> I_k = new ArrayList<Box>();
////						for(int i=0;i<sortedYZBox.size();i++) {
////							if(sortedYZBox.get(i).getXCoor()+sortedYZBox.get(i).getWidth()>yz_levels.get(k)) {
////								I_k.add(new Box(sortedYZBox.get(i)));
////							}
////						}
////						//求2DCorners==========================================================begin
//////						Corners2Dx = new ArrayList<Double>();
//////						Corners2Dy = new ArrayList<Double>();
////						if(I_k.size()<1) {
//////							Corners2Dx.add(0.0);Corners2Dy.add(0.0);
////							if(!sumWHL.contains(yz_levels.get(k))) {
////								Box corner = new Box(zerobox);
////								corner.setXCoor(yz_levels.get(k));corner.setYCoor(0.0);corner.setZCoor(0.0);
////								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////							}//else {
////								
////							//}
////						}else {
////							//Phase 1: identify the extreme items e_1,...,e_m
////							ArrayList<Integer> e = new ArrayList<Integer>();
////							double bar_z = 0.0;//注意I_k是根据y,x排序的。
////							for(int i=0;i<I_k.size();i++) {
////								if(I_k.get(i).getZCoor()+I_k.get(i).getLength()>bar_z) {
////									e.add(i);bar_z=I_k.get(i).getZCoor()+I_k.get(i).getLength();//
////								}
////							}
////							//Phase 2: determine the corner points
////							double ZCoor = 0.0;
////							double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
////							if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
//////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
////								if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
////									Box corner = new Box(zerobox);
////									corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
////									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////								}
////							}
////							for(int j=1;j<e.size();j++) {
////								ZCoor = I_k.get(e.get(j-1)).getZCoor()+I_k.get(e.get(j-1)).getLength();
////								YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
////								if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
//////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
////									if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
////										Box corner = new Box(zerobox);
////										corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
////										Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////									}
////								}
////							}
////							ZCoor = I_k.get(e.get(e.size()-1)).getZCoor()+I_k.get(e.get(e.size()-1)).getLength();
////							YCoor = 0.0;
////							if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
//////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
////								if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
////								Box corner = new Box(zerobox);
////								corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
////								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////								}
////							}
////						}
////						//找完了这一层的2DCorners==========================================================end
//////						for(int i=0;i<Corners2Dx.size();i++) {
////////							if(Corners2DxLast.contains(Corners2Dx.get(i))&&Corners2DyLast.contains(Corners2Dy.get(i))) {//如果上一层已经有这个corner了
////////								//如果x和y都有的话，看他们的index是否相同。
////////								if(Corners2DxLast.indexOf(Corners2Dx.get(i))!=Corners2DyLast.indexOf(Corners2Dy.get(i))){
////////									Box corner = new Box(zerobox);
////////									corner.setXCoor(Corners2Dx.get(i));
////////									corner.setYCoor(Corners2Dy.get(i));
////////									corner.setZCoor(horizontal_levels.get(k));
////////									Corners3D.add(corner);
////////								}
////////							}else {//如果上一层没有这个corner，则增加这个corner。
//////								Box corner = new Box(zerobox);
//////								corner.setXCoor(Corners2Dx.get(i));
//////								corner.setYCoor(Corners2Dy.get(i));
//////								corner.setZCoor(horizontal_levels.get(k));
//////								Corners3D.add(corner);
////////							}
//////						}
////						//save this Corners2D
//////						Corners2DxLast = (ArrayList<Double>)Corners2Dx.clone();
//////						Corners2DyLast = (ArrayList<Double>)Corners2Dy.clone();
//////						k=k+1;
////					}//k
//					
//					
//					
//					}// if I
//					//第一步先求3DCorners=========================================================end
//					//找一个位置去存放当前箱子boxingSeqence.get(boxi)。
//					//将这些位置随机打乱。
////					int [] permutation= com.my.vrp.utils.Permutation.intPermutation(Corners3D.size());
////					将这些位置进行排序，bottom-left->
//					//原来的排序是Z->X->Y
//					//尝试按x轴进行排序。
//					Corners3D.sort(new java.util.Comparator<Box>() {
//						@Override
//						public int compare(Box o1, Box o2) {
//							// TODO Auto-generated method stub
//							if(o1.getZCoor()<o2.getZCoor()) {
//								return -1;
//							} else if (o1.getZCoor()>o2.getZCoor()) {
//								return 1;
//							} else {
//								if(o1.getXCoor()<o2.getXCoor())
//									return -1;
//								else if(o1.getXCoor()>o2.getXCoor())
//									return 1;
//								else {
//									if (o1.getYCoor()<o2.getYCoor())
//										return -1;
//									else if(o1.getYCoor()>o2.getYCoor())
//										return 1;
//									else
//										return 0;
//								}
//							}
//						}
//					});
//					for(int positioni=0;positioni<Corners3D.size();positioni++) {
//						Box curr_position = Corners3D.get(positioni);
//						if(curr_position.getXCoor()+curr_box.getWidth()<=this.width&&
//								curr_position.getYCoor()+curr_box.getHeight()<=this.height&&
//										curr_position.getZCoor()+curr_box.getLength()<=this.length) {
//							//判断这个位置能不能站稳
//							//当前箱子的坐标： boxingSequence.x,y,z
//							//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//							//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//							boolean support = false;
//							if(curr_position.getYCoor()==0) {
//								support = true;
//							}else{
//								//计算该箱子的底部面积。
////								Box currBox = boxingSequence.get(i);
//								double bottomArea = curr_box.getWidth()*curr_box.getLength();
//								double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
//								double crossArea = 0;
//								//计算所有已放箱子的顶部与该箱子的底部交叉面积
////								boolean flagXYcross = false;
//								for (int boxii=0;boxii<thisBoxes.size();boxii++) {
//									//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//									Box existBox = thisBoxes.get(boxii);
//									
//									if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=0.001) {
//										double xc=curr_position.getXCoor(),yc=curr_position.getYCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ye=existBox.getYCoor(),ze=existBox.getZCoor();
//										double wc=curr_box.getWidth(),hc=curr_box.getHeight(),lc=curr_box.getLength(),we=existBox.getWidth(),he=existBox.getHeight(),le=existBox.getLength();
//										
//										if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
//											double [] XCoor = {xc,xc+wc,xe,xe+we};
//											double [] ZCoor = {zc,zc+lc,ze,ze+le};
//											//sort xc,xc+wc,xe,xe+we
//											 Arrays.sort(XCoor);
//											 Arrays.sort(ZCoor);
//											//sort zc,zc+lc,ze,ze+le
//											 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//											 if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%并且已经有交叉了，则不用继续判断了。
//										}
////										if((xc+wc>xe)&&(ze+le>zc)) {
////											crossArea = crossArea+Math.min(xc+wc-xe,wc)*Math.min(ze+le-zc,lc);
////										}
////										if((xe+we>xc)&&(zc+lc>ze)) {
////											crossArea = crossArea+Math.min(xe+we-xc,wc)*Math.min(zc+lc-ze,lc);
////										}
////										//判断当前箱子和已经存在的箱子在X-Y平面是否有交叉。如果已经有交叉，则不用继续判断了。如果没有交叉则需要继续判断，直到最后。
////										这里没有用着,2020年11月20日，失败的一次。
////										if(!((xc+wc<xe)||(xe+we<xc)||(yc+hc<ye)||(ye+he<yc))) {
////											flagXYcross = true;
////										}
//									}
//								}
//								
//							}
//							//
//							if(support) {
//								Box loadBox = new Box(curr_box);
//								loadBox.setXCoor(curr_position.getXCoor());
//								loadBox.setYCoor(curr_position.getYCoor());
//								loadBox.setZCoor(curr_position.getZCoor());
//								
//								
//								
//								
//								//插入到sortedYXBox
//								int idx=0;
//								for(idx=0;idx<sortedYXBox.size();idx++) {//按y,x来排序。
//									Box thisbox = sortedYXBox.get(idx);
//									//如果在一个水平面上，则对比X
//									if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<10) {
//										if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
//									}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//										break;
//									}
////									if(&&Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())>10) {
////										//对Y进行排序
////										break;
////									}else if(&&) {//对X进行排序。
////										break;
////									}//else if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<0.01&&
//									//		Math.abs(thisbox.getXCoor()+thisbox.getWidth()-loadBox.getXCoor()-loadBox.getWidth())<0.01&&
//									//		thisbox.getZCoor()+this.getLength()>loadBox.getZCoor()+loadBox.getLength()) {
//									//	break;
//									//}
//								}
//								if(idx==sortedYXBox.size()) {
//									sortedYXBox.add(new Box(loadBox));//sort sortedBox by y,x coordinate
//								}else {
//									sortedYXBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//								}
//								//插入到yx_levels
//								double curr_level = loadBox.getZCoor()+loadBox.getLength();
//								if(!yx_levels.contains(curr_level)) {
//									
//									for(idx=0;idx<yx_levels.size();idx++) {
//										if(yx_levels.get(idx)>curr_level) {
//											break;
//										}
//									}
//									if(idx==yx_levels.size()) {
//										yx_levels.add(curr_level);
//									}else {
//										yx_levels.add(idx, curr_level);
//									}
//								}
//								
//								
//								
//								
//								//插入到sortedYZBox
//								idx=0;
//								for(idx=0;idx<sortedYZBox.size();idx++) {//按Y,Z来排序。
//									Box thisbox = sortedYZBox.get(idx);
//									//如果在一个平面上，则对比Z
//									if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<10) {
//										if(thisbox.getZCoor()+thisbox.getLength()<loadBox.getZCoor()+loadBox.getLength()) {break;}
//									}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//										break;
//									}
////									if(&&Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())>10) {
////										//对Y进行排序
////										break;
////									}else if(&&) {//对X进行排序。
////										break;
////									}//else if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<0.01&&
//									//		Math.abs(thisbox.getXCoor()+thisbox.getWidth()-loadBox.getXCoor()-loadBox.getWidth())<0.01&&
//									//		thisbox.getZCoor()+this.getLength()>loadBox.getZCoor()+loadBox.getLength()) {
//									//	break;
//									//}
//								}
//								if(idx==sortedYZBox.size()) {
//									sortedYZBox.add(new Box(loadBox));//sort sortedBox by y,x coordinate
//								}else {
//									sortedYZBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//								}
//								//插入到yz_levels
//								curr_level = loadBox.getXCoor()+loadBox.getWidth();
//								if(!yz_levels.contains(curr_level)) {
//									
//									for(idx=0;idx<yz_levels.size();idx++) {
//										if(yz_levels.get(idx)>curr_level) {
//											break;
//										}
//									}
//									if(idx==yz_levels.size()) {
//										yz_levels.add(curr_level);
//									}else {
//										yz_levels.add(idx, curr_level);
//									}
//								}
//								
//								
//								//if this horizontal level loadBox.y+loadBox.height not in horizontal_levels, then add it and sort horizontal levels by increasing values
//								
//								thisBoxes.add(loadBox);
//								thisloadWeight = thisloadWeight + loadBox.getWeight();
//								loadIdx.add(boxi);//第i-2+1个箱子被装载了。
//								insertConfirm=true;
//								break;
//							}
//						}
//					}
//					if(!insertConfirm) {break;}
//				}
//				
//				this.Boxes = thisBoxes;
//				this.loadWeight=thisloadWeight;
//				//calculate excessWeight
//				if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//				//calculate excessLength.
////				if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
////						+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
////					this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
////							+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
////				else
//					this.excessLength = 0;
////				System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////				if(left.leftSequence.size()<boxingSequence1.size())
////				System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//				return loadIdx;//left.leftSequence.size();
//			}
		
	
	
			/**
			 * dblf算法装箱,根据yx，yz进行排序找3DCorners
			 * 其他的三个策略：YZ plane的Corners3D的检查，每次检查当前level之前是否有更合适的（应该是不好，这样会使得很不整齐，是否对3D corners进行排序。
			 * @param clients
			 * @return 装箱的box下标数组,装箱个数
			 */
//				public ArrayList<Integer> dblf3(ArrayList<Box> boxingSequence) {
//					boolean YZPlane=false;
//					boolean checkLevel=false;
//					boolean sortCorners3D=false;
//					//boxingSequence是请求放在当前小车的箱子序列，每个平台的箱子从大到小排序。
//					ArrayList<Double> yx_levels = new ArrayList<Double>();
//					yx_levels.add(0.0);
//					ArrayList<Double> yz_levels = new ArrayList<Double>();
//					yz_levels.add(0.0);
//					ArrayList<Box> sortedYXBox = new ArrayList<Box>();//清空已经存放的boxes
//					ArrayList<Box> sortedYZBox = new ArrayList<Box>();//清空已经存放的boxes
//					ArrayList<Box> thisBoxes = new ArrayList<Box>();//
//					double thisloadWeight=0.0;
//					ArrayList<Integer> loadIdx=new ArrayList<Integer>();
//					Box zerobox = new Box();
//					zerobox.setHeight(0);
//					zerobox.setLength(0);
//					zerobox.setWidth(0);
//					zerobox.setXCoor(0);
//					zerobox.setYCoor(0);
//					zerobox.setZCoor(0);
//					zerobox.setDirection(100);
//					boolean insertConfirm;//是否成功插入当前箱子。
//					for(int boxi=0;boxi<boxingSequence.size();boxi++) {
//						insertConfirm=false;
//						Box curr_box = boxingSequence.get(boxi);
//						if(thisloadWeight + curr_box.getWeight()>this.capacity) {
//							break;//推出寻找最优位置，并且退出遍历箱子。
//						}
//						//第一步先求3DCorners=========================================================
//						ArrayList<Box> Corners3D = new ArrayList<Box>();//如果已经存放的箱子是0，则原点。
//						ArrayList<Double> sumWHL = new ArrayList<Double>();
//						
//						
//						//对YX平面。
//						if(sortedYXBox.size()<1) {
//							Corners3D.add(new Box(zerobox));sumWHL.add(0.0);
//						}
////						else if (sortedBox.size()<2){
////							if(sortedBox.get(0).getHeight()+curr_box.getHeight()<=this.height) {
//////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
////								Box corner = new Box(zerobox);
////								corner.setXCoor(0.0);corner.setYCoor(sortedBox.get(0).getHeight());corner.setZCoor(0.0);Corners3D.add(corner);
////							}
////							if(sortedBox.get(0).getWidth()+curr_box.getWidth()<=this.width) {
////							Box corner = new Box(zerobox);
////							corner.setXCoor(sortedBox.get(0).getWidth());corner.setYCoor(0.0);corner.setZCoor(0.0);Corners3D.add(corner);
////							}
////							Box corner = new Box(zerobox);
////							corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(sortedBox.get(0).getLength());Corners3D.add(corner);
//////							for(int j=1;j<e.size();j++) {
//////								XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//////								YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//////								if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//////									Box corner = new Box(zerobox);
//////									corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));Corners3D.add(corner);
//////								}
//////							}
//////							XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//////							YCoor = 0.0;
//////							if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////////								Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//////								Box corner = new Box(zerobox);
//////								corner.setXCoor(XCoor);corner.setYCoor(YCoor);corner.setZCoor(horizontal_levels.get(k));Corners3D.add(corner);
//////							}
//////							Corners3D.add();
////						}
//						else {
//							
////							ArrayList<Double> Corners2DxLast = new ArrayList<Double>();
////							ArrayList<Double> Corners2DyLast = new ArrayList<Double>();
////							ArrayList<Double> Corners2Dx;
////							ArrayList<Double> Corners2Dy;
////						int k=0;//遍历每个Z平面，和Z轴length垂直的平面。
//						//保存每一层所有的箱子。
//						ArrayList<ArrayList<Box>> I_K=new ArrayList<ArrayList<Box>>();//保存倒数第二个平面上的箱子。用来处理最后一个。
//						for(int k=0;k<yx_levels.size() && yx_levels.get(k)+curr_box.getLength()<=this.length;k++) {
//							//得到在这个平面之上的已经存放的boxes
//							ArrayList<Box> I_k = new ArrayList<Box>();
//							for(int i=0;i<sortedYXBox.size();i++) {
//								if(sortedYXBox.get(i).getZCoor()+sortedYXBox.get(i).getLength()>yx_levels.get(k)) {
//									I_k.add(new Box(sortedYXBox.get(i)));
//								}
//							}
////							if(k==yx_levels.size()-2) {
////								
////							}
//							//求2DCorners==========================================================begin
////							Corners2Dx = new ArrayList<Double>();
////							Corners2Dy = new ArrayList<Double>();
//							if(I_k.size()<1) {
////								Corners2Dx.add(0.0);Corners2Dy.add(0.0);
////								if(k>=1&&k==yx_levels.size()-1) {//如果是最后一层，检查是否可以向前推进。
////									//x=curr_box.getWidth()<all lastbutoneBoxes.x //yx_levels.get(k)
////									int formeri=0;
////									for(formeri=0;formeri<lastbutoneBoxes.size();formeri++) {
////										if(!(curr_box.getWidth()<=lastbutoneBoxes.get(formeri).getXCoor())) {
////											break;//找到了这种的话，提前结束。
////										}
////									}
////									double thislevel = yx_levels.get(k);
////									if(formeri==lastbutoneBoxes.size()) {
////										//不存在会和这个box交叉的箱子。
////										//将这个点往前移动一个level
////										thislevel = yx_levels.get(k-1);
////									}
////									Box corner = new Box(zerobox);
////									corner.setXCoor(0.0);corner.setYCoor(0.0);corner.setZCoor(thislevel);
////									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////								}else {
//								double XCoor = 0.0;
//								double YCoor = 0.0;
//								Box corner = new Box(zerobox);
//								corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//								//find a level for ZCoor, k,k-1,k-2
//								int currLevel = k;
////								boolean levelFound=false;
//								if(checkLevel)
//								for(int currk=k-1;currk>=0;currk--) {
//									ArrayList<Box> I = I_K.get(currk);
//									int i=0;
//									for(i=0;i<I.size();i++) {
//										if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//											//this box should parallel with new box.
//											if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//												break;//找到了这种的话，提前结束。
//											}
//										}
//									}
//									if(i==I.size()) {currLevel = currk;}else{break;}
//								}
//								corner.setZCoor(yx_levels.get(currLevel));
//								Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
////								}
//							}else {
//								//Phase 1: identify the extreme items e_1,...,e_m
//								ArrayList<Integer> e = new ArrayList<Integer>();
//								double bar_x = 0.0;//注意I_k是根据y,x排序的。
//								for(int i=0;i<I_k.size();i++) {
//									if(I_k.get(i).getXCoor()+I_k.get(i).getWidth()>bar_x) {
//										e.add(i);bar_x=I_k.get(i).getXCoor()+I_k.get(i).getWidth();//
//									}
//								}
//								//Phase 2: determine the corner points
//								double XCoor = 0.0;
//								double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//								if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									Box corner = new Box(zerobox);
//									corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//									//find a level for ZCoor, k,k-1,k-2
//									int currLevel = k;
////									boolean levelFound=false;
//									if(checkLevel)
//									for(int currk=k-1;currk>=0;currk--) {
//										ArrayList<Box> I = I_K.get(currk);
//										int i=0;
//										for(i=0;i<I.size();i++) {
//											if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//												//this box should parallel with new box.
//												if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//													break;//找到了这种的话，提前结束。
//												}
//											}
//										}
//										if(i==I.size()) {currLevel = currk;}else{break;}
//									}
//									corner.setZCoor(yx_levels.get(currLevel));
//									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//								}
//								for(int j=1;j<e.size();j++) {
//									XCoor = I_k.get(e.get(j-1)).getXCoor()+I_k.get(e.get(j-1)).getWidth();
//									YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//									if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////										Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//										Box corner = new Box(zerobox);
//										corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//										//find a level for ZCoor, k,k-1,k-2
//										int currLevel = k;
////										boolean levelFound=false;
//										if(checkLevel)
//										for(int currk=k-1;currk>=0;currk--) {
//											ArrayList<Box> I = I_K.get(currk);
//											int i=0;
//											for(i=0;i<I.size();i++) {
//												if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//													//this box should parallel with new box.
//													if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//														break;//找到了这种的话，提前结束。
//													}
//												}
//											}
//											if(i==I.size()) {currLevel = currk;}else{break;}
//										}
//										corner.setZCoor(yx_levels.get(currLevel));
//										Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//									}
//								}
//								XCoor = I_k.get(e.get(e.size()-1)).getXCoor()+I_k.get(e.get(e.size()-1)).getWidth();
//								YCoor = 0.0;
//								if(XCoor+curr_box.getWidth()<=this.width&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									Box corner = new Box(zerobox);
//									corner.setXCoor(XCoor);corner.setYCoor(YCoor);
//									//find a level for ZCoor, k,k-1,k-2
//									int currLevel = k;
////									boolean levelFound=false;
//									if(checkLevel)
//									for(int currk=k-1;currk>=0;currk--) {
//										ArrayList<Box> I = I_K.get(currk);
//										int i=0;
//										for(i=0;i<I.size();i++) {
//											if(Math.abs(I.get(i).getYCoor()+I.get(i).getHeight()-YCoor)>50) {//this box should at this y-level.
//												//this box should parallel with new box.
//												if(!(XCoor+curr_box.getWidth()<=I.get(i).getXCoor()||(XCoor>=I.get(i).getXCoor()+I.get(i).getWidth()))) {
//													break;//找到了这种的话，提前结束。
//												}
//											}
//										}
//										if(i==I.size()) {currLevel = currk;}else{break;}
//									}
//									corner.setZCoor(yx_levels.get(currLevel));
//									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//								}
//							}
//							//保存这一层的箱子。
//							if(checkLevel)
//							I_K.add(I_k);
//						}//k
//						//重新调整所有的Corners3D,看yx-level是否可以往前移动。
//						
//						//找yz这一层的。
//						if(YZPlane)
//						for(int k=0;k<yz_levels.size() && yz_levels.get(k)+curr_box.getWidth()<=this.width;k++) {
//							//得到在这个平面之上的已经存放的boxes
//							ArrayList<Box> I_k = new ArrayList<Box>();
//							for(int i=0;i<sortedYZBox.size();i++) {
//								if(sortedYZBox.get(i).getXCoor()+sortedYZBox.get(i).getWidth()>yz_levels.get(k)) {
//									I_k.add(new Box(sortedYZBox.get(i)));
//								}
//							}
//							//求2DCorners==========================================================begin
////							Corners2Dx = new ArrayList<Double>();
////							Corners2Dy = new ArrayList<Double>();
//							if(I_k.size()<1) {
////								Corners2Dx.add(0.0);Corners2Dy.add(0.0);
//								if(!sumWHL.contains(yz_levels.get(k))) {
//									Box corner = new Box(zerobox);
//									corner.setXCoor(yz_levels.get(k));corner.setYCoor(0.0);corner.setZCoor(0.0);
//									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//								}//else {
//									
//								//}
//							}else {
//								//Phase 1: identify the extreme items e_1,...,e_m
//								ArrayList<Integer> e = new ArrayList<Integer>();
//								double bar_z = 0.0;//注意I_k是根据y,x排序的。
//								for(int i=0;i<I_k.size();i++) {
//									if(I_k.get(i).getZCoor()+I_k.get(i).getLength()>bar_z) {
//										e.add(i);bar_z=I_k.get(i).getZCoor()+I_k.get(i).getLength();//
//									}
//								}
//								//Phase 2: determine the corner points
//								double ZCoor = 0.0;
//								double YCoor = I_k.get(e.get(0)).getYCoor()+I_k.get(e.get(0)).getHeight();
//								if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
//										Box corner = new Box(zerobox);
//										corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
//										Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//									}
//								}
//								for(int j=1;j<e.size();j++) {
//									ZCoor = I_k.get(e.get(j-1)).getZCoor()+I_k.get(e.get(j-1)).getLength();
//									YCoor = I_k.get(e.get(j)).getYCoor()+I_k.get(e.get(j)).getHeight();
//									if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
////										Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//										if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
//											Box corner = new Box(zerobox);
//											corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
//											Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//										}
//									}
//								}
//								ZCoor = I_k.get(e.get(e.size()-1)).getZCoor()+I_k.get(e.get(e.size()-1)).getLength();
//								YCoor = 0.0;
//								if(ZCoor+curr_box.getLength()<=this.length&&YCoor+curr_box.getHeight()<=this.height) {
////									Corners2Dx.add(XCoor);Corners2Dy.add(YCoor);
//									if(!sumWHL.contains(yz_levels.get(k)+YCoor+ZCoor)) {
//									Box corner = new Box(zerobox);
//									corner.setXCoor(yz_levels.get(k));corner.setYCoor(YCoor);corner.setZCoor(ZCoor);
//									Corners3D.add(corner);sumWHL.add(corner.getXCoor()+corner.getYCoor()+corner.getZCoor());
//									}
//								}
//							}
//							//找完了这一层的2DCorners==========================================================end
////							for(int i=0;i<Corners2Dx.size();i++) {
//////								if(Corners2DxLast.contains(Corners2Dx.get(i))&&Corners2DyLast.contains(Corners2Dy.get(i))) {//如果上一层已经有这个corner了
//////									//如果x和y都有的话，看他们的index是否相同。
//////									if(Corners2DxLast.indexOf(Corners2Dx.get(i))!=Corners2DyLast.indexOf(Corners2Dy.get(i))){
//////										Box corner = new Box(zerobox);
//////										corner.setXCoor(Corners2Dx.get(i));
//////										corner.setYCoor(Corners2Dy.get(i));
//////										corner.setZCoor(horizontal_levels.get(k));
//////										Corners3D.add(corner);
//////									}
//////								}else {//如果上一层没有这个corner，则增加这个corner。
////									Box corner = new Box(zerobox);
////									corner.setXCoor(Corners2Dx.get(i));
////									corner.setYCoor(Corners2Dy.get(i));
////									corner.setZCoor(horizontal_levels.get(k));
////									Corners3D.add(corner);
//////								}
////							}
//							//save this Corners2D
////							Corners2DxLast = (ArrayList<Double>)Corners2Dx.clone();
////							Corners2DyLast = (ArrayList<Double>)Corners2Dy.clone();
////							k=k+1;
//						}//k
//						
//						
//						
//						}// if I
//						//第一步先求3DCorners=========================================================end
//						//找一个位置去存放当前箱子boxingSeqence.get(boxi)。
//						//将这些位置随机打乱。
////						int [] permutation= com.my.vrp.utils.Permutation.intPermutation(Corners3D.size());
////						将这些位置进行排序，bottom-left->
//						//原来的排序是Z->X->Y
//						//尝试按x轴进行排序。
//						if(sortCorners3D)
//						Corners3D.sort(new java.util.Comparator<Box>() {
//							@Override
//							public int compare(Box o1, Box o2) {
//								// TODO Auto-generated method stub
//								if(o1.getZCoor()<o2.getZCoor()) {
//									return -1;
//								} else if (o1.getZCoor()>o2.getZCoor()) {
//									return 1;
//								} else {
//									if(o1.getXCoor()<o2.getXCoor())
//										return -1;
//									else if(o1.getXCoor()>o2.getXCoor())
//										return 1;
//									else {
//										if (o1.getYCoor()<o2.getYCoor())
//											return -1;
//										else if(o1.getYCoor()>o2.getYCoor())
//											return 1;
//										else
//											return 0;
//									}
//								}
//							}
//						});
//						for(int positioni=0;positioni<Corners3D.size();positioni++) {
//							Box curr_position = Corners3D.get(positioni);
//							if(curr_position.getXCoor()+curr_box.getWidth()<=this.width&&
//									curr_position.getYCoor()+curr_box.getHeight()<=this.height&&
//											curr_position.getZCoor()+curr_box.getLength()<=this.length) {
//								//判断这个位置能不能站稳
//								//当前箱子的坐标： boxingSequence.x,y,z
//								//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//								//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//								boolean support = false;
//								if(curr_position.getYCoor()==0) {
//									support = true;
//								}else{
//									//计算该箱子的底部面积。
////									Box currBox = boxingSequence.get(i);
//									double bottomArea = curr_box.getWidth()*curr_box.getLength();
//									double curr_y = curr_position.getYCoor();//+boxingSequence.get(i).getHeight();
//									double crossArea = 0;
//									//计算所有已放箱子的顶部与该箱子的底部交叉面积
////									boolean flagXYcross = false;
//									for (int boxii=0;boxii<thisBoxes.size();boxii++) {
//										//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//										Box existBox = thisBoxes.get(boxii);
//										
//										if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=0.001) {
//											double xc=curr_position.getXCoor(),yc=curr_position.getYCoor(),zc=curr_position.getZCoor(),xe=existBox.getXCoor(),ye=existBox.getYCoor(),ze=existBox.getZCoor();
//											double wc=curr_box.getWidth(),hc=curr_box.getHeight(),lc=curr_box.getLength(),we=existBox.getWidth(),he=existBox.getHeight(),le=existBox.getLength();
//											
//											if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {//如果有交叉，则计算交叉面积。
//												double [] XCoor = {xc,xc+wc,xe,xe+we};
//												double [] ZCoor = {zc,zc+lc,ze,ze+le};
//												//sort xc,xc+wc,xe,xe+we
//												 Arrays.sort(XCoor);
//												 Arrays.sort(ZCoor);
//												//sort zc,zc+lc,ze,ze+le
//												 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//												 if(crossArea>=0.8*bottomArea) {support=true;break;}//如果支撑面积大于80%并且已经有交叉了，则不用继续判断了。
//											}
////											if((xc+wc>xe)&&(ze+le>zc)) {
////												crossArea = crossArea+Math.min(xc+wc-xe,wc)*Math.min(ze+le-zc,lc);
////											}
////											if((xe+we>xc)&&(zc+lc>ze)) {
////												crossArea = crossArea+Math.min(xe+we-xc,wc)*Math.min(zc+lc-ze,lc);
////											}
////											//判断当前箱子和已经存在的箱子在X-Y平面是否有交叉。如果已经有交叉，则不用继续判断了。如果没有交叉则需要继续判断，直到最后。
////											这里没有用着,2020年11月20日，失败的一次。
////											if(!((xc+wc<xe)||(xe+we<xc)||(yc+hc<ye)||(ye+he<yc))) {
////												flagXYcross = true;
////											}
//										}
//									}
//									
//								}
//								//
//								if(support) {
//									Box loadBox = new Box(curr_box);
//									loadBox.setXCoor(curr_position.getXCoor());
//									loadBox.setYCoor(curr_position.getYCoor());
//									loadBox.setZCoor(curr_position.getZCoor());
//									
//									
//									
//									
//									//插入到sortedYXBox
//									int idx=0;
//									for(idx=0;idx<sortedYXBox.size();idx++) {//按y,x来排序。
//										Box thisbox = sortedYXBox.get(idx);
//										//如果在一个水平面上，则对比X
//										if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<10) {
//											if(thisbox.getXCoor()+thisbox.getWidth()<loadBox.getXCoor()+loadBox.getWidth()) {break;}
//										}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//											break;
//										}
////										if(&&Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())>10) {
////											//对Y进行排序
////											break;
////										}else if(&&) {//对X进行排序。
////											break;
////										}//else if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<0.01&&
//										//		Math.abs(thisbox.getXCoor()+thisbox.getWidth()-loadBox.getXCoor()-loadBox.getWidth())<0.01&&
//										//		thisbox.getZCoor()+this.getLength()>loadBox.getZCoor()+loadBox.getLength()) {
//										//	break;
//										//}
//									}
//									if(idx==sortedYXBox.size()) {
//										sortedYXBox.add(new Box(loadBox));//sort sortedBox by y,x coordinate
//									}else {
//										sortedYXBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//									}
//									//插入到yx_levels
//									double curr_level = loadBox.getZCoor()+loadBox.getLength();
//									if(!yx_levels.contains(curr_level)) {
//										
//										for(idx=0;idx<yx_levels.size();idx++) {
//											if(yx_levels.get(idx)>curr_level) {
//												break;
//											}
//										}
//										if(idx==yx_levels.size()) {
//											yx_levels.add(curr_level);
//										}else {
//											yx_levels.add(idx, curr_level);
//										}
//									}
//									
//									
//									
//									if(YZPlane) {
//									//插入到sortedYZBox
//									idx=0;
//									for(idx=0;idx<sortedYZBox.size();idx++) {//按Y,Z来排序。
//										Box thisbox = sortedYZBox.get(idx);
//										//如果在一个平面上，则对比Z
//										if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<10) {
//											if(thisbox.getZCoor()+thisbox.getLength()<loadBox.getZCoor()+loadBox.getLength()) {break;}
//										}else if(thisbox.getYCoor()+thisbox.getHeight()<loadBox.getYCoor()+loadBox.getHeight()){//如果不在一个水平面上，则对比Y
//											break;
//										}
////										if(&&Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())>10) {
////											//对Y进行排序
////											break;
////										}else if(&&) {//对X进行排序。
////											break;
////										}//else if(Math.abs(thisbox.getYCoor()+thisbox.getHeight()-loadBox.getYCoor()-loadBox.getHeight())<0.01&&
//										//		Math.abs(thisbox.getXCoor()+thisbox.getWidth()-loadBox.getXCoor()-loadBox.getWidth())<0.01&&
//										//		thisbox.getZCoor()+this.getLength()>loadBox.getZCoor()+loadBox.getLength()) {
//										//	break;
//										//}
//									}
//									if(idx==sortedYZBox.size()) {
//										sortedYZBox.add(new Box(loadBox));//sort sortedBox by y,x coordinate
//									}else {
//										sortedYZBox.add(idx, new Box(loadBox));//sort sortedBox by y,x coordinate
//									}
//									//插入到yz_levels
//									curr_level = loadBox.getXCoor()+loadBox.getWidth();
//									if(!yz_levels.contains(curr_level)) {
//										
//										for(idx=0;idx<yz_levels.size();idx++) {
//											if(yz_levels.get(idx)>curr_level) {
//												break;
//											}
//										}
//										if(idx==yz_levels.size()) {
//											yz_levels.add(curr_level);
//										}else {
//											yz_levels.add(idx, curr_level);
//										}
//									}
//									}
//									
//									//if this horizontal level loadBox.y+loadBox.height not in horizontal_levels, then add it and sort horizontal levels by increasing values
//									
//									thisBoxes.add(loadBox);
//									thisloadWeight = thisloadWeight + loadBox.getWeight();
//									loadIdx.add(boxi);//第i-2+1个箱子被装载了。
//									insertConfirm=true;
//									break;
//								}
//							}
//						}
//						if(!insertConfirm) {break;}
//					}
//					
//					this.Boxes = thisBoxes;
//					this.loadWeight=thisloadWeight;
//					//calculate excessWeight
//					if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//					//calculate excessLength.
////					if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
////							+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
////						this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
////								+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
////					else
//						this.excessLength = 0;
////					System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////					if(left.leftSequence.size()<boxingSequence1.size())
////					System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//					return loadIdx;//left.leftSequence.size();
//				}
				
	
	
	
	
	
	
//	public ArrayList<Integer> dblfxxx(ArrayList<Box> boxingSequence1) {
//		ArrayList<Integer> loadIdx=new ArrayList<Integer>();
//		ArrayList<Box> loadBoxes = new ArrayList<Box>();
//		this.loadWeight=0;
//		//********* insert two empty box to boxingSequence
//		ArrayList<Box> boxingSequence = new ArrayList<Box>();
//		Box boxTop = new Box();
//		Box boxBack = new Box();
////		boxTop.setNumber(0);
//		boxTop.setHeight(0);
//		boxTop.setLength(0);
//		boxTop.setWidth(0);
//		boxTop.setXCoor(0);
//		boxTop.setYCoor(0);
//		boxTop.setZCoor(0);
////		boxBack.setNumber(-1);
//		boxBack.setHeight(0);
//		boxBack.setLength(0);
//		boxBack.setWidth(0);
//		boxBack.setXCoor(0);
//		boxBack.setYCoor(0);
//		boxBack.setZCoor(0);
//		boxingSequence.add(boxBack);
//		boxingSequence.add(boxTop);
//		for (Box box : boxingSequence1) {
//			Box box1 = new Box(box);
////			box1.setHeight(box.getHeight());
////			box1.setLength(box.getLength());
//////			box1.setNumber(box.getNumber());
////			box1.setWidth(box.getWidth());
////			box1.setXCoor(box.getXCoor());
////			box1.setYCoor(box.getYCoor());
////			box1.setZCoor(box.getZCoor());
//			boxingSequence.add(box1);
//		}
//		//***********************插入两个0箱子
////		double loaded_weights = 0;
//		LeftSequence left = new LeftSequence();
//		BackSequence back = new BackSequence();
//		TopSequence top = new TopSequence();
//		//在back和top各插入一个0箱子
//		back.backSequence.add(boxingSequence.get(0));
//		top.topSequence.add(boxingSequence.get(1));
//		
//		//在原点插入第一个箱子
//		back.backSequence.add(boxingSequence.get(2));this.loadWeight=this.loadWeight+boxingSequence.get(2).getWeight();
//		left.leftSequence.add(boxingSequence.get(2));
//		loadBoxes.add(new Box(boxingSequence.get(2)));
//		top.topSequence.add(boxingSequence.get(2));
//		loadIdx.add(0);//第一个箱子被装载了。
//		//计算其他箱子的坐标。
//		for(int i=3;i<boxingSequence.size();i++) {
//			boolean insertConfirm = false;//begin to insert i-th box, insertConfirm to indicate whether this insertion is successful.
//			for(int j=0;j<back.backSequence.size();j++) {
//				if(back.backSequence.get(j).getZCoor()+back.backSequence.get(j).getLength()+boxingSequence.get(i).getLength()<=length) {
//					boxingSequence.get(i).setZCoor(back.backSequence.get(j).getZCoor()+back.backSequence.get(j).getLength());
//					//表示在backBox[j].Z+backBox[j].length这个坐标有空间来存放当前box[i]
//				}
//				else continue;
//				for(int k=0;k<top.topSequence.size();k++) {
//					if(top.topSequence.get(k).getYCoor()+top.topSequence.get(k).getHeight()+boxingSequence.get(i).getHeight()<=height) {
//						boxingSequence.get(i).setYCoor(top.topSequence.get(k).getYCoor()+top.topSequence.get(k).getHeight());
//						//表示在topBox[k].Y+topBox[k].height这个坐标平面有空间来存放当前box[i]
//					}
//					else continue;
//					boxingSequence.get(i).setXCoor(0);
//					for(int p = 0;p<=left.leftSequence.size();p++) {
//						//遍历所有已经存放的box
//						boolean flag = false;
//						if(p>left.leftSequence.size()-1||boxingSequence.get(i).getXCoor()+boxingSequence.get(i).getWidth()>width)
//							flag = true;
//						else if(boxingSequence.get(i).getXCoor()+boxingSequence.get(i).getWidth()<=left.leftSequence.get(p).getXCoor())
//							flag = true;
//						else if(boxingSequence.get(i).getXCoor()<left.leftSequence.get(p).getXCoor()+left.leftSequence.get(p).getWidth()
//								&&boxingSequence.get(i).getYCoor()<left.leftSequence.get(p).getYCoor()+left.leftSequence.get(p).getHeight()
//								&&boxingSequence.get(i).getZCoor()<left.leftSequence.get(p).getZCoor()+left.leftSequence.get(p).getLength()
//								)
//							boxingSequence.get(i).setXCoor(left.leftSequence.get(p).getXCoor()+left.leftSequence.get(p).getWidth());
//
//						if(flag==true) {
//							
//							if(boxingSequence.get(i).getXCoor()+boxingSequence.get(i).getWidth()<=width) {
//								//判断这个位置能不能站稳
//								//当前箱子的坐标： boxingSequence.x,y,z
//								//当前箱子的底部高度：boxingSequence.y，如果为0的话，就可以了
//								//遍历所有的已经放了的箱子，看是否支撑现在的箱子。（暴力了点）
//								boolean support = false;
//								if(boxingSequence.get(i).getYCoor()==0) {
//									support = true;
//								}else{
//									//计算该箱子的底部面积。
//									Box currBox = boxingSequence.get(i);
//									double bottomArea = currBox.getWidth()*currBox.getLength();
//									double curr_y = currBox.getYCoor();//+boxingSequence.get(i).getHeight();
//									double crossArea = 0;
//									//计算所有已放箱子的顶部与该箱子的底部交叉面积
//									for (int boxi=0;boxi<left.leftSequence.size();boxi++) {
//										//如果这个箱子的顶部与boxingSequence.get(i)的底部在同一水平上
//										Box existBox = left.leftSequence.get(boxi);
//										if(Math.abs(existBox.getYCoor()+existBox.getHeight()-curr_y)<=0.001) {
//											double xc=currBox.getXCoor(),zc=currBox.getZCoor(),xe=existBox.getXCoor(),ze=existBox.getZCoor();
//											double wc=currBox.getWidth(),lc=currBox.getLength(),we=existBox.getWidth(),le=existBox.getLength();
//											
//											if(!((xc+wc<xe)||(xe+we<xc)||(zc+lc<ze)||(ze+le<zc))) {
//												double [] XCoor = {xc,xc+wc,xe,xe+we};
//												double [] ZCoor = {zc,zc+lc,ze,ze+le};
//												//sort xc,xc+wc,xe,xe+we
//												 Arrays.sort(XCoor);
//												 Arrays.sort(ZCoor);
//												//sort zc,zc+lc,ze,ze+le
//												 crossArea = crossArea + Math.abs(XCoor[2]-XCoor[1])*Math.abs(ZCoor[2]-ZCoor[1]);
//												 if(crossArea>0.8*bottomArea) {support=true;break;}
//											}
////											if((xc+wc>xe)&&(ze+le>zc)) {
////												crossArea = crossArea+Math.min(xc+wc-xe,wc)*Math.min(ze+le-zc,lc);
////											}
////											if((xe+we>xc)&&(zc+lc>ze)) {
////												crossArea = crossArea+Math.min(xe+we-xc,wc)*Math.min(zc+lc-ze,lc);
////											}
//											//判断交叉面积。
//											
//										}
//									}
//									
//								}
//								//
//								if(support) {
//									left.leftSequence.add(boxingSequence.get(i));
//									loadBoxes.add(new Box(boxingSequence.get(i)));
//									this.loadWeight = this.loadWeight + boxingSequence.get(i).getWeight();
//									
//									back.backSequence.add(boxingSequence.get(i));//x-y plane
//									top.topSequence.add(boxingSequence.get(i));//x-z plane
//									
////									left.leftSort();
////									back.backSort();
////									top.topSort();
//									insertConfirm=true;
//									loadIdx.add(i-2);//第i-2+1个箱子被装载了。
//								}
//							}
//							break;
//						}
//					}
//					if(insertConfirm)
//						break;
//				}
//				if(insertConfirm)
//					break;
//			}
//		}
//		this.Boxes = left.leftSequence;
//		//calculate excessWeight
//		if(this.loadWeight>this.capacity) {this.excessWeight=this.loadWeight-this.capacity;}else {this.excessWeight=0;}
//		//calculate excessLength.
//		if(back.backSequence.get(back.backSequence.size()-1).getZCoor()
//				+back.backSequence.get(back.backSequence.size()-1).getLength()>this.length)
//			this.excessLength = back.backSequence.get(back.backSequence.size()-1).getZCoor()
//					+back.backSequence.get(back.backSequence.size()-1).getLength()-this.length;
//		else
//			this.excessLength = 0;
////		System.out.println("excessLength:"+this.excessLength+";excessWeight:"+this.excessWeight);
////		if(left.leftSequence.size()<boxingSequence1.size())
////		System.out.println("input box size:"+boxingSequence1.size()+"this vehicle size:"+this.Boxes.size());
//		return loadIdx;//left.leftSequence.size();
//	}



//public void setVolume(double volume) {
//	this.volume = volume;
//}
//public static void main() {
	
//}
}
