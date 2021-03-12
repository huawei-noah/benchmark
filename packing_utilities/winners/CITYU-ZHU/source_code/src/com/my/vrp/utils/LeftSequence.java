package com.my.vrp.utils;

import java.util.ArrayList;
import java.util.Collections;

import com.my.vrp.Box;
/**
 * left序列类
 * @author dell
 *
 */
public class LeftSequence {

	/**
	 * 存放left序列的箱子
	 * 适用于DBLF算法
	 */
	public ArrayList<Box> leftSequence = new ArrayList<Box>();
	/**
	 * 为left序列排序
	 */
	public  void leftSort() {
		for(int i=0;i<this.leftSequence.size()-1;i++) {
			for(int j=0;j<this.leftSequence.size()-i-1;j++) {
				if(this.leftSequence.get(j).getXCoor()>this.leftSequence.get(j+1).getXCoor()) {
					Collections.swap(this.leftSequence, j, j+1);
				}
			}
		}
	}
}
