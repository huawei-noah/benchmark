package com.my.vrp.SSBH;

import java.util.ArrayList;
import java.util.LinkedList;

import static java.lang.Math.abs;

public class SimpleSpace {
    private LinkedList<Space> space_list;
    public SimpleSpace(LinkedList<Space> space_list){
        this.space_list = space_list;
    }

    public void update_space_list(SimpleBlock block_use, Space space_using) {
        LinkedList<Space> temp = new LinkedList<Space>();
        //按照长宽高的顺序来处理
        Double mz = space_using.getLength() - block_use.getLength();
        Double mx = space_using.getWidth() - block_use.getWidth();

        Double[] Coords = space_using.getCoord();
        Double[] Coords_z_space = Coords.clone();
        Coords_z_space[0] += block_use.getLength();

        Double[] Coords_x_space = Coords.clone();
        Coords_x_space[1] += block_use.getWidth();

        Double[] Coords_y_space = Coords.clone();
        Coords_y_space[2] += block_use.getHeight();

        Double[] Coords_trans_space = Coords.clone();
        Coords_trans_space[0] += block_use.getLength();
        Coords_trans_space[1] += block_use.getWidth();

        Space space_trans = new Space(mz, mx, space_using.getHeight(), Coords_trans_space);

        if(mz>mx) {
            Space container_z = new Space(
                    mz,
                    space_using.getWidth(),
                    space_using.getHeight(),
                    Coords_z_space,
                    space_trans,
                    space_using.getHold_surface()
            );
            Space container_x = new Space(
                    block_use.getLength(),
                    mx,
                    space_using.getHeight(),
                    Coords_x_space,
                    space_using.getHold_surface()
            );
            temp.add(container_x);
            temp.add(container_z);
        }else{
            Space container_z = new Space(
                    mz,
                    block_use.getWidth(),
                    space_using.getHeight(),
                    Coords_z_space,
                    space_using.getHold_surface()
            );
            Space container_x = new Space(
                    space_using.getLength(),
                    mx,
                    space_using.getHeight(),
                    Coords_x_space,
                    space_trans,
                    space_using.getHold_surface()
            );
            temp.add(container_z);
            temp.add(container_x);
        }
        temp.add(new Space(
                block_use.getLength(),
                block_use.getWidth(),
                space_using.getHeight() - block_use.getHeight(),
                Coords_y_space,
                block_use.getHold_surface()
        ));
        this.space_list.addAll(temp);
    }

    public void trans_space(Space space_using) {
        /* TODO */
        Space trans_space = space_using.getTrans_space();
        if(trans_space!=null){
            Space next_space = this.space_list.removeLast();
            Double[] Coords_trans = trans_space.getCoord();
            Double[] Coords_next = next_space.getCoord();
            if(abs(Coords_trans[0]-Coords_next[0]) < 0.01){
                next_space.setWidth(next_space.getWidth()+trans_space.getWidth());
            }else{
                next_space.setLength(next_space.getLength()+trans_space.getLength());
            }
            space_list.add(next_space);
        }
    }
}
