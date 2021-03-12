package com.my.vrp;

public class EP {
	private double XCoor;
	private double YCoor;
	private double ZCoor;
	private double length;
	private double width;
	private double height;
	private double bottom_area;
	private int platformID;//这个位置可以存放哪个平台的box
	public EP() {
		platformID=0;
	}
	public EP(EP copyEP) {
		this.XCoor = copyEP.getXCoor();
		this.YCoor = copyEP.getYCoor();
		this.ZCoor = copyEP.getZCoor();
		this.length = copyEP.getLength();
		this.width = copyEP.getWidth();
		this.height = copyEP.getHeight();
		this.bottom_area = copyEP.getBottom_area();
		this.platformID = copyEP.getPlatformID();
	}
//	public double getBottomArea() {
//		return this.length*this.width;
//	}
	public double getXCoor() {
		return XCoor;
	}
	public void setXCoor(double xCoor) {
		XCoor = xCoor;
	}
	public double getYCoor() {
		return YCoor;
	}
	public void setYCoor(double yCoor) {
		YCoor = yCoor;
	}
	public double getZCoor() {
		return ZCoor;
	}
	public void setZCoor(double zCoor) {
		ZCoor = zCoor;
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
	public int getPlatformID() {
		return platformID;
	}
	public void setPlatformID(int platformID) {
		this.platformID = platformID;
	}
	public double getBottom_area() {
		return bottom_area;
	}
	public void setBottom_area(double bottom_area) {
		this.bottom_area = bottom_area;
	}
}
