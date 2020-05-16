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


//#include "App.h"
#include "String.h"
#include <omnetpp.h>
#include <stdio.h>
#include "..\Messages\Packet_m.h"
using namespace omnetpp;

class eApp:public cSimpleModule{
  private:
    // configuration
    int myAddress;

    int n; //number of all nodes
    cPar *sendIATime;// time between generating packets
    cPar *packetLengthBytes;// length of one message (fixed! no "volatile" modifier)
    cPar * pattern;
    // state
    cMessage *generatePacket;
    int pkCounter;

//  public:


  protected:
    virtual void initialize() ;
    virtual void handleMessage(cMessage *msg) ;
    virtual void finish();

    cStdDev hopStats;
    cStdDev eedStats;
    cStdDev throughput;
};
Define_Module(eApp);


void eApp::initialize(){
    //hopCountSignal = registerSignal("hopCounter");
    hopStats.setName("hops");
    eedStats.setName("End-to-End Delay");
    throughput.setName("throughput");

    myAddress = ((int)getParentModule()->getParentModule()->par("erackid"))*10+(int)getParentModule()->par("id");//Node's address
    n = 160;//network nodes number
    pattern = &par("pattern");
        packetLengthBytes = &par("packetLength");
        sendIATime = &par("sendIaTime");  // volatile parameter
        pkCounter = 0;

        WATCH(pkCounter);
        WATCH(myAddress);

        generatePacket = new cMessage("nextPacket");
        scheduleAt(sendIATime->doubleValue(), generatePacket);//0.0
}
void eApp::handleMessage(cMessage *msg){
    if (msg == generatePacket) {
            // Sending packet
        int destAddress;
        if(pattern->doubleValue() == (double)1){
            //Pattern1: uniform
            destAddress = intuniform(0, n-1);
            while(destAddress == myAddress){
                destAddress = intuniform(0,n-1);
            }
        }
        else if(pattern->doubleValue() == (double)2){
            //Pattren2: neighbor-1
            int prob = intuniform(0, 1);
            if(prob == 0){
                destAddress = (myAddress-1+n) % n;
            }
            else{
                destAddress = (myAddress+1) % n;
            }
        }
        else if(pattern->doubleValue() == (double)3){

            //Pattren3: neighbor-2
            int prob = intuniform(0, 9);
            if(prob == 0 || prob == 1 || prob == 2 ||prob == 3){
                destAddress = (myAddress-1+n) % n;
            }
            else if(prob == 4){
                destAddress = (myAddress-2+n) % n;
            }
            else if(prob == 5 || prob == 6 || prob == 7 ||prob == 8){
                destAddress = (myAddress+1) % n;
            }
            else if(prob == 9){
                destAddress = (myAddress+2) % n;
            }
        }
        else{
            EV<<"Error! pattern can not be recognized! "<<endl;
        }

            char pkname[40];
            sprintf(pkname, "myAddress-%d-to-%d", myAddress, destAddress);

            EV << "generating packet " << pkname << endl;

            Packet *pk = new Packet(pkname);
            pk->setByteLength(packetLengthBytes->longValue());
            pk->setSrcAddr(myAddress);
            pk->setDestAddr(destAddress);
            //pk->setHopCount(pkCounter);//多加了一句
            send(pk, "out");

            scheduleAt(simTime() + sendIATime->doubleValue(), generatePacket);

    }
    else {
            // Handle incoming packet
        if(dynamic_cast <Packet *> (msg) != NULL){
            Packet *pk = check_and_cast<Packet *>(msg);

            pkCounter++;
            throughput.collect(pkCounter);

            int hopCount = pk->getHopCount()+1;
            EV << "eApp Address: "<< myAddress << " received packet " << pk->getName() << " after " << hopCount << " hops" << endl;
            hopStats.collect(hopCount);

            simtime_t eed = simTime() - pk->getCreationTime();
            eedStats.collect(eed);
            delete pk;
        }
    }

}
void eApp::finish(){
      throughput.record();//throughput max即为收包总数
      eedStats.record();
      hopStats.record();
}


