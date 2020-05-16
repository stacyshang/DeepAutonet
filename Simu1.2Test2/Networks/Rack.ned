
package Networks;
import Nodes.Node;
import Switches.*;

module Rack
{
    parameters:
        int rackid;
        int noodenum = 10;
        
        @display("bgb=208.01001,164.70001");
    gates:
        input i[2+2];//electriccal (neighbor)+optical(ocsnum)
        output o[2+2];//neighbor+ocsnum

//        input opti[ocsnum];//optical inter-connection
//        output opto[ocsnum];

        input routingctrlin;
        output routingctrlout;

    submodules:
         
        node[noodenum]: Node {              
            id = index;
            @display("p=87.23,132.98001");
        }

        eps: EPS {

            @display("p=126.270004,56.730003");
        }
    connections:
        for i=0..noodenum-1 {
            eps.port$o[i] --> halfEPSdelay --> node[i].appin;
            eps.port$i[i] <-- halfEPSdelay <-- node[i].appout;
        }

        for j=0..2+2-1 {
            eps.port$o[j+noodenum] --> o[j];
            eps.port$i[j+noodenum] <-- i[j];
        }

        eps.routingctrlout --> routingctrlout;
        eps.routingctrlin <-- routingctrlin;
}
