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

//#include "OCS.h"
#include <map>
//#include "Routing.h"
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"
#include "..\Messages\CtrltoOCSTopoAdd_m.h"
#include "..\Messages\CtrltoOCSTopoCut_m.h"
using namespace omnetpp;

class OCS : public cSimpleModule{
    private:
//       int myAddress;
//       std::map<int,int> OCSmap ;
       int* schedulingTable;

    protected:
       virtual void initialize();
       virtual void handleMessage(cMessage *msg);
       virtual void setGateConnect(int inGate,int outGate);
       virtual void unsetGateConnect(int inGate,int outGate);
//       virtual bool ifConnected(int x,int y);//whether port x is connected to port y
};
Define_Module(OCS);

void OCS::initialize(){
//    myAddress = (int)getModule()->par("ocsid")+100000;

    schedulingTable = (int*)calloc(gateSize("in"),sizeof(int));//在内存的动态存储区中分配n个长度为size的连续空间，函数返回一个指向分配起始地址的指针
    for(int i=0;i<gateSize("in");i++){
          schedulingTable[i] = -1;
          WATCH(schedulingTable[i]);
    }

}
void OCS::handleMessage(cMessage *msg){
    if(dynamic_cast <Packet *> (msg) != NULL){
//    Packet* s = dynamic_cast<Packet* >(msg);
//    if(s){
       Packet *pkt = check_and_cast<Packet *>(msg);
       cGate *gate = pkt->getArrivalGate();
       if(schedulingTable[gate->getIndex()] == -1)
           delete pkt;   // Output gate not assigned. Drop msg
       else{
           pkt->setHopCount(pkt->getHopCount()+1);   //HopCount+1
           send(pkt,"out",schedulingTable[gate->getIndex()]);
       }
       EV<<"OCS sending pkt out...."<<endl;
    }
    else if(dynamic_cast <CtrltoOCSTopoAdd *> (msg) != NULL){
//    CtrltoOCSTopoAdd* t = dynamic_cast<CtrltoOCSTopoAdd* >(msg);
//    if(t){
        CtrltoOCSTopoAdd *info = check_and_cast<CtrltoOCSTopoAdd*>(msg);
        this->setGateConnect(info->getInGate(), info->getOutGate());//"this"???
    }
    else if(dynamic_cast <CtrltoOCSTopoCut *> (msg) != NULL){
//    CtrltoOCSTopoCut* r = dynamic_cast<CtrltoOCSTopoCut* >(msg);
//    if(r){
        CtrltoOCSTopoCut *info = check_and_cast<CtrltoOCSTopoCut*>(msg);
        this->unsetGateConnect(info->getInGate(), info->getOutGate());
    }
}
void OCS::setGateConnect(int inGate,int outGate){
   if(schedulingTable[inGate] != -1) EV << "Attempting to schedule an already scheduled input channel. Channel id: %d" << inGate << endl;
   schedulingTable[inGate] = outGate;
}
void OCS::unsetGateConnect(int inGate,int outGate){
    if (schedulingTable[inGate] == outGate){
            schedulingTable[inGate] = -1;
    }
}
//OCS::OCS() {
//    // TODO Auto-generated constructor stub
//}
//
//OCS::~OCS() {
//    // TODO Auto-generated destructor stub
//    free(schedulingTable);
//}

