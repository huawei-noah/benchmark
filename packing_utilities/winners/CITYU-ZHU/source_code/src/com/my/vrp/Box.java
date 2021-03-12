package com.my.vrp;
/**
 * 货物箱子类
 * @author dell
 *
 */
@SuppressWarnings("rawtypes")
public class Box implements Comparable ,Cloneable{
	private int id;
	private String spuBoxID;
	private double length;
	private double width;
	private double height;
	private double weight;
	private int direction;
	private double XCoor;
	private double YCoor;
	private double ZCoor;
	private int platformid;

	public Box() {
		
	}
	public Box(Box next) {
		// TODO Auto-generated constructor stub
		this.spuBoxID=next.spuBoxID;
		this.length=next.length;
		this.width=next.width;
		this.height=next.height;
		this.weight=next.weight;
		this.direction=next.direction;
		this.XCoor=next.XCoor;
		this.YCoor=next.YCoor;
		this.ZCoor=next.ZCoor;
		this.platformid=next.platformid;
		this.id = next.id;
	}
	public String getSpuBoxID() {
		return spuBoxID;
	}
	public void setSpuBoxID(String spuBoxID) {
		this.spuBoxID = spuBoxID;
	}
	
	public double getLength() {
		if(direction==100) {
			return length;
		}else {
			return width;
		}
	}
	public void setLength(double length) {
		this.length = length;
	}
	public double getWidth() {
		if(direction==100) {
			return width;
		}else if(direction==200){
			return length;
		}else {
			System.out.println("error direction of box: "+this.spuBoxID);
			return 0.0;
		}
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
	//direction
	public double getDirection() {
		return direction;
	}
	public void setDirection(int direction) {
		this.direction = direction;
	}
	//XCoor
	public double getXCoor() {
		return XCoor;
	}
	public void setXCoor(double xCoor) {
		this.XCoor = xCoor;
	}
	public double getYCoor() {
		return YCoor;
	}
	public void setYCoor(double yCoor) {
		this.YCoor = yCoor;
	}
	public double getZCoor() {
		return ZCoor;
	}
	public void setZCoor(double zCoor) {
		this.ZCoor = zCoor;
	}
	@Override
	public String toString() {
		return "Box [spuBoxID=" + spuBoxID + ", length=" + length + ", width=" + width + ", height=" + height + ", XCoor="
				+ XCoor + ", YCoor=" + YCoor + ", ZCoor=" + ZCoor + "]";
	}
	@Override
	public int compareTo(Object o) {
		// TODO Auto-generated method stub
		Box box = (Box) o;
		//sort by weight
//		double volume1 = this.weight;
//		double volume2 = box.weight;
		
//		double volume1=this.height*this.length*this.width;
//		double volume2=box.height*box.length*box.width;
		
		double volume1=this.length*this.width;
		double volume2=box.length*box.width;
		
//		double volume1=this.length;
//		double volume2=box.length;
//		double volume1=this.height;
//		double volume2=box.height;
		
		//bottom area first, height second
//		if(volume1>volume2)
//			return -1;
//		else if (volume1<volume2)
//			return 1;
//		else
//			if(this.height>box.height)
//				return -1;
//			else if(this.height<box.height)
//				return 1;
//			else
//				return 0;
		
		//height first, bottom area second、
		if(this.height>box.height)
			return -1;
		else if(this.height<box.height)
			return 1;
		else
			if(volume1>volume2)
				return -1;
			else if (volume1<volume2)
				return 1;
			else
				return 0;
		
		//small2large
//		if(volume1<volume2)
//			return -1;
//		else if(volume1==volume2)
//			return 0;
//		else
//			return 1;
		
	}
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		Box box = null;
		try {
			box = (Box)super.clone();
		}catch(CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return box;
	}
	public double getWeight() {
		return weight;
	}
	public void setWeight(double weight) {
		this.weight = weight;
	}
	public int getPlatformid() {
		return platformid;
	}
	public void setPlatformid(int platformid) {
		this.platformid = platformid;
	}
	
	public double getVolume() {
		return this.height*this.length*this.width;
	}
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	
}
