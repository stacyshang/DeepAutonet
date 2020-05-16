//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package Switches;

module OCSch
{
    parameters:
        //     int ocsid;//address
        //    @unit;
        int racknum = 16;
        @display("bgb=256,228");
    gates:
        input in[racknum];
        output out[racknum];
        input ctrlin;
        output ctrlout;
    submodules:
        checkBox[racknum]: ocs2rCheckBox {
            @display("p=110,38");
        }
        ocs:OCS{
        }
    connections:
        for i=0..racknum-1 {
            in[i] --> ocs.in[i];
            ocs.out[i] --> checkBox[i].in;
            checkBox[i].line --> out[i];
        }
        ctrlin --> ocs.ctrlin;
        ocs.ctrlout --> ctrlout;

}
simple OCS
{
    parameters:
//     int ocsid;//address
//    @unit;
       int racknum=16;
    gates:
        input in[racknum];
        output out[racknum];
        input ctrlin;
        output ctrlout;
}

simple ocs2rCheckBox
{
    parameters:
        //int Capacity ; // max number of packets; 0 means no limit
        //bool useCutThroughSwitching = default(false);  // use cut-through switching instead of store-and-forward
        //@display("i=block/queue;q=queue");

    gates:
        input in;
//        output out;
//        inout line;
        output line;
}
