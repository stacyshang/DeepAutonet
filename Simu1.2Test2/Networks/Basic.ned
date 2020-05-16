
package Networks;
import Nodes.Node;
import Switches.*;
//channel
channel EPSdelay extends ned.DatarateChannel
{
    delay = 400 ns;
    datarate = default(1Gbps);
}
channel halfEPSdelay extends ned.DatarateChannel
{
    delay = 200 ns;
    //datarate = default(1Gbps);
}
channel halfOCSdelay extends ned.DatarateChannel
{
    delay = 20 ns;
    datarate = default(40Gbps);
}

network Basic
{
    parameters:
        int NumNodes = 160;
        int racknum = 16;
        int ocsnum = 2;
        int neighbor = 2;
        @display("bgb=414,292");
    gates:

    submodules:
        rack[racknum]: Rack {
            rackid = index;        
            @display("p=277,92");
        }
        
        ocsch[ocsnum]: OCSch {
 //           ocsid = index;
            @display("p=162,177");
        }
        controller: Controller {
            @display("p=30,69");

        }
    connections:
        for i=0..racknum-1 {
////            rack[(i-1+racknum)%racknum].o[1] --> rack[i].i[0];
////            rack[i].o[0] --> rack[(i+1)%racknum].i[1];

            rack[i].o[1] --> EPSdelay --> rack[(i+1)%racknum].i[0];
            rack[i].i[1] <-- EPSdelay <-- rack[(i+1)%racknum].o[0];
        }

        
        for i=0..racknum-1 {
            rack[i].o[2] --> halfOCSdelay --> ocsch[0].in[i];
            rack[i].i[2] <-- halfOCSdelay <-- ocsch[0].out[i];
        } 
        for i=0..racknum-1 {
            rack[i].o[3] --> halfOCSdelay --> ocsch[1].in[i];
            rack[i].i[3] <-- halfOCSdelay <-- ocsch[1].out[i];
        }


        for i=0..ocsnum-1 {
            controller.ocsout[i] --> ocsch[i].ctrlin;
            controller.ocsin[i] <-- ocsch[i].ctrlout;
        }
        for i=0..racknum-1 {
            controller.routingout[i] --> rack[i].routingctrlin;
            controller.routingin[i] <-- rack[i].routingctrlout;
        }
}

