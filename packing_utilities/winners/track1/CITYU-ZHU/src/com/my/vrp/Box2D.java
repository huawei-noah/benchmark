package com.my.vrp;

import java.util.ArrayList;
import java.util.Iterator;

public class Box2D {
	//
	private ArrayList<Box> Boxes;
	private double length;//max(all)
	private double width;//max(all)
	private double height;//
	private double weight;//
	private int direction;//
	private double XCoor;
	private double YCoor;
	private double ZCoor;//
	
	public Box2D() {
		this.Boxes = new ArrayList<Box>();
		this.length=0.0;
		this.width = 0.0;
		this.height = 0.0;
		this.weight=0.0;
		this.direction=100;
		this.XCoor=0.0;
		this.YCoor=0.0;
		this.ZCoor=0.0;
	}
	
	public Box2D(Box2D copy) {
		this.Boxes = new ArrayList<Box>();
		Iterator<Box> i = copy.getBoxes().iterator();
		while(i.hasNext()) this.Boxes.add(new Box(i.next()));
		this.length = copy.getLength();
		this.width = copy.getWidth();
		this.height = copy.getHeight();
		this.weight = copy.getWeight();
		this.direction = copy.getDirection();
		this.XCoor = copy.getXCoor();
		this.YCoor = copy.getYCoor();
		this.ZCoor = copy.getZCoor();
	}

	public ArrayList<Box> getBoxes() {
		return Boxes;
	}

	public void setBoxes(ArrayList<Box> boxes) {
		Boxes = boxes;
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

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public int getDirection() {
		return direction;
	}

	public void setDirection(int direction) {
		this.direction = direction;
	}

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
}
