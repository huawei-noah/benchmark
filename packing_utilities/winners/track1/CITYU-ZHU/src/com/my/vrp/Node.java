package com.my.vrp;

import java.util.ArrayList;
import java.util.Iterator;
/**
 * 结点类<br/>
 * number:客户编号<br/>
 * XCoor:X坐标<br/>
 * YCoor:Y坐标<br/>
 * goodsNum:货物数量<br/>
 * demands:需求货物的重量<br/>
 * beginTime:时间窗最早到达时间<br/>
 * endTime:时间窗最晚到达时间<br/>
 * serviceTime:服务时间<br/>
 * goods:货物信息<br/>
 * @author dell
 *
 */
public  class Node implements Cloneable,Comparable<Node>{
	private int platformid;//id of platform, 用做distance matrix的下标，从1到n
//	private double XCoor;
//	private double YCoor;
	private int goodsNum;
	private double goodsWeight;
	private double goodsVolumn;
	private int loadgoodsNum;
	/**
	 * 这个变量在route的binpacking里面用来记录这个节点在这条路径的这个车里面的箱子长度。
	 */
	private double demands;//这个变量在route的binpacking里面用来记录这个节点在这条路径的这个车里面的箱子长度。
	private double weights;//这个变量保存这个节点所有箱子的重量。
	private boolean mustFirst;
//	private String platform;
//	private double beginTime;//时间窗最早达到时间
//	private double endTime;//最晚到达时间
//	private double serviceTime;//服务时间
	private double reachTime;
	private ArrayList<Box> goods = new ArrayList<Box>();//客户的货物

	public Node() {
		demands = 0.0;
		weights = 0.0;
	}
	public Node(Node copyNode) {
		demands = copyNode.demands;
		weights = copyNode.weights;
		platformid = copyNode.getPlatformID();
		goodsNum = copyNode.getGoodsNum();
		loadgoodsNum = copyNode.getLoadgoodsNum();
		goodsWeight = copyNode.goodsWeight;
		goodsVolumn = copyNode.goodsVolumn;
		mustFirst = copyNode.isMustFirst();
		goods = new ArrayList<Box>();
		Iterator<Box> iterator = copyNode.getGoods().iterator();
		while(iterator.hasNext()) goods.add(new Box(iterator.next()));
	}
//	public double getXCoor() {
//		return XCoor;
//	}
//	public void setXCoor(double xCoor) {
//		XCoor = xCoor;
//	}
//	public double getYCoor() {
//		return YCoor;
//	}
//	public void setYCoor(double yCoor) {
//		YCoor = yCoor;
//	}
	public int getPlatformID() {
		return platformid;
	}
	public void setPlatformID(int platformid) {
		this.platformid = platformid;
	}
	public int getGoodsNum() {
		return goodsNum;
	}
	public void setGoodsNum(int goodsNum) {
		this.goodsNum = goodsNum;
	}
	public double getDemands() {
		return demands;
	}
	public boolean isMustFirst() {
		return mustFirst;
	}
	public void setMustFirst(boolean mustFirst) {
		this.mustFirst = mustFirst;
	}
	public void setDemands(double demands) {
		this.demands = demands;
	}
//	public double getBeginTime() {
//		return beginTime;
//	}
//	public void setBeginTime(double beginTime) {
//		this.beginTime = beginTime;
//	}
//	public double getEndTime() {
//		return endTime;
//	}
//	public void setEndTime(double endTime) {
//		this.endTime = endTime;
//	}
//	public double getServiceTime() {
//		return serviceTime;
//	}
//	public void setServiceTime(double serviceTime) {
//		this.serviceTime = serviceTime;
//	}
	
	public double getReachTime() {
		return reachTime;
	}
	public void setReachTime(double reachTime) {
		this.reachTime = reachTime;
	}
	public ArrayList<Box> getGoods() {
		return goods;
	}
	public void setGoods(ArrayList<Box> goods) {
		this.goods = goods;
	}
	
	@Override
	public String toString() {
	return "Node [number=" + platformid + ", goodsNum=" + goodsNum
			+ ", demands=" + demands + ", goods=" + goods + "]";
}
//	public String toString() {
//		return "Node [number=" + number + ", XCoor=" + XCoor + ", YCoor=" + YCoor + ", goodsNum=" + goodsNum
//				+ ", demands=" + demands + ", beginTime=" + beginTime + ", endTime=" + endTime + ", serviceTime="
//				+ serviceTime + ", reachTime=" + reachTime + ", goods=" + goods + "]";
//	}
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		Node node = null;
		try {
			node = (Node)super.clone();
		}catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		ArrayList<Box> boxes = new ArrayList<Box>();
		Iterator<Box> iterator = goods.iterator();
		while(iterator.hasNext()) {
			boxes.add((Box) iterator.next().clone());
		}
		node.setGoods(boxes);
		return node;
	}
	@Override
	public int compareTo(Node o) {
		// TODO Auto-generated method stub
		if (this.platformid>o.platformid) {
			return 1;
		}else if(this.platformid<o.platformid) {
			return -1;
		}
		return 0;
	}
	public int getLoadgoodsNum() {
		return loadgoodsNum;
	}
	public void setLoadgoodsNum(int loadgoodsNum) {
		this.loadgoodsNum = loadgoodsNum;
	}
	public double getWeights() {
		return weights;
	}
	public void setWeights(double weights) {
		this.weights = weights;
	}
	public double getGoodsWeight() {
		return goodsWeight;
	}
	public void setGoodsWeight(double goodsWeight) {
		this.goodsWeight = goodsWeight;
	}
	public double getGoodsVolumn() {
		return goodsVolumn;
	}
	public void setGoodsVolumn(double goodsVolumn) {
		this.goodsVolumn = goodsVolumn;
	}
	
}
