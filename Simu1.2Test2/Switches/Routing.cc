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

#include <map>
//#include "Routing.h"
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"
#include "..\Messages\CtrltoRoutingTableMod_m.h"
using namespace omnetpp;

class Routing : public cSimpleModule{
    private:
       int myAddress;

       typedef std::map<int,int> RoutingTable; // destaddr -> gateindex
       RoutingTable rtable;
       std::map<int,int> ocs0map,ocs1map;

       int neighbor = 2;
       int Nodenum = 10;

//       simsignal_t dropSignal;
//       simsignal_t outputIfSignal;
    protected:
       virtual void initialize();
       virtual void handleMessage(cMessage *msg);
};
Define_Module(Routing);

void Routing::initialize(){
    myAddress = (int)getParentModule()->getParentModule()->par("rackid")+10000;

    //initially establish routingtable when not receiving OCS connecting
    cTopology *topo = new cTopology("topo");
    topo->extractByProperty("unit");//Node+EPS, no OCS
    cTopology ::Node *thisNode = topo->getNodeFor(getParentModule());
//
    thisNode->setWeight((double)0);
//
    EV << "cTopology found" << topo->getNumNodes() << " (no OCS) nodes\n";

        // find and store next hops
        for (int i = 0; i < topo->getNumNodes(); i++) {
            if (topo->getNode(i) == thisNode)
                continue;  // skip ourselves

            topo->calculateUnweightedSingleShortestPathsTo(topo->getNode(i));

            if (thisNode->getNumPaths() == 0)
                continue;  // not connected

            cGate *parentModuleGate = thisNode->getPath(0)->getLocalGate();
            int gateIndex = parentModuleGate->getIndex();
            int address;
            //if it's Node
            if(topo->getNode(i)->getModule()->getModuleType() != thisNode->getModule()->getModuleType()){//不知道有没有问题！！
                address= 10*(int)topo->getNode(i)->getModule()->getParentModule()->par("rackid")+(int)topo->getNode(i)->getModule()->par("id");

            }   //is EPS
            else if(topo->getNode(i)->getModule()->getModuleType() == thisNode->getModule()->getModuleType()){
                address = 10000+(int)topo->getNode(i)->getModule()->getParentModule()->par("rackid");
            }
            else{
                address = -1;
                EV << " node module type is illegal" << endl;
            }
            rtable[address] = gateIndex;
            //EV << "  towards address " << address << " gateIndex is " << gateIndex << endl;
        }
        delete topo;

}
void Routing::handleMessage(cMessage *msg){

    //packet
    if(dynamic_cast <Packet *> (msg) != NULL){

         Packet *pk = check_and_cast<Packet *>(msg);
         int destAddr = pk->getDestAddr();
         //int srcAddr = pk->getSrcAddr();

         //dest is a routing module (which is impossible)
         if (destAddr == myAddress) {
             EV << "discarding illegal local routing delivery of packet " << pk->getName() << endl;
             delete pk;
             //emit(outputIfSignal, -1);  // -1: local
             return;
         }

         RoutingTable::iterator it = rtable.find(destAddr);
         if (it == rtable.end()) {
             EV << "address " << destAddr << " unreachable, discarding packet " << pk->getName() << endl;
             //emit(dropSignal, (long)pk->getByteLength());
             delete pk;
             return;
         }
         int outGateIndex = (*it).second;
         EV << "forwarding packet " << pk->getName() << " on gate index " << outGateIndex << endl;
         pk->setHopCount(pk->getHopCount()+1);
         //emit(outputIfSignal, outGateIndex);
         send(pk, "out", outGateIndex);
    }

    //ocsmap from controller(we can calculate routingtable)
    else if(dynamic_cast <CtrltoRoutingTableMod *> (msg) != NULL){

        //recover routing table
        ocs0map.clear();ocs1map.clear();

        CtrltoRoutingTableMod *mod = check_and_cast<CtrltoRoutingTableMod *>(msg);
        for (int i=0;i<mod->getOcs0mapsize();i++){
            ocs0map.insert(std::map<int,int>::value_type(mod->getOcs0mapleft(i),mod->getOcs0mapright(i)));
        }
        for (int j=0;j<mod->getOcs1mapsize();j++){
            ocs1map.insert(std::map<int,int>::value_type(mod->getOcs1mapleft(j),mod->getOcs1mapright(j)));
        }
        EV << "mod->getOcs0mapsize(): "<<mod->getOcs0mapsize()<< ", mod->getOcs1mapsize(): "<<mod->getOcs1mapsize()<<endl;
        EV << "routing establishes ocs maps, ocsomap & ocs1map's size is " <<ocs0map.size()<<","<<ocs1map.size()<< endl;
        //calculate routing table (to be sent to each Routing)

        // find and store next hops
        cTopology *_topo = new cTopology("_topo");
        _topo->extractByProperty("unit");//new _topo (add some nodes and links)
        cTopology ::Node *target = _topo->getNodeFor(getParentModule());
        for(int q=0;q<_topo->getNumNodes();q++){
            _topo->getNode(q)->setWeight((double)0);
        }
//
        for (int j = 0; j < _topo->getNumNodes(); j++) {
            for (int i = 0; i < _topo->getNode(j)->getNumInLinks(); i++) {
                _topo->getNode(j)->getLinkIn(i)->setWeight((double)2);
            }
        }
//
            //Add OCS Nodes
            //int previousNodeNum = (int)_topo->getNumNodes();
            std::map<int,int>::iterator i;
            cTopology ::Node *OCS0[100];cTopology::Link* link0a[200];cTopology::Link* link0b[200];
            int x = 0;
            for(i=ocs0map.begin(); i!=ocs0map.end(); i++){//_topo->getNumNodes() = 176 now(no added ocs nodes)
                //注意看看！！
                OCS0[x]= new cTopology ::Node;
                OCS0[x]->setWeight((double)1);
                int index = _topo->addNode(OCS0[x]);//Returns the index of the new graph node* (see getNode(int))

                int h,k;
                for(int q=0;q<(int)_topo->getNumNodes();q++){//topo->getNodes(i) is EPS(not Node) & rackid is in ocsmap
                    if(_topo->getNode(q)->getModule()->getModuleType() == target->getModule()->getModuleType() && i->first == (int)_topo->getNode(q)->getModule()->getParentModule()->par("rackid")){
                        h = q;
                        break;
                    }
                }
                for(int q=0;q<(int)_topo->getNumNodes();q++){
                    if(_topo->getNode(q)->getModule()->getModuleType() == target->getModule()->getModuleType() && i->second == (int)_topo->getNode(q)->getModule()->getParentModule()->par("rackid")){
                        k = q;
                        break;
                    }
                }
                EV<<"OCS0 find corresponding node id: h = "<<h<<", k = "<<k<<endl;

                link0a[x] = new cTopology::Link;
                link0b[x] = new cTopology::Link;
//
                link0a[x]->setWeight(1);
                link0b[x]->setWeight(1);
//
                _topo->addLink(link0a[x], _topo->getNode(h), _topo->getNode(index));//不知道是不是单向连接？
                _topo->addLink(link0b[x], _topo->getNode(index), _topo->getNode(k));
                x++;
            }
            //EV<<"ocs0index:　"<<ocs0index[0]<<" "<<ocs0index[1]<<" "<<ocs0index[2]<<endl;
            //int _previousNodeNum = (int)_topo->getNumNodes();
            std::map<int,int>::iterator j;
            cTopology ::Node *OCS1[100];cTopology::Link* link1a[200];cTopology::Link* link1b[200];
            int xx = 0;
            for(j=ocs1map.begin(); j!=ocs1map.end(); j++){
                OCS1[xx]= new cTopology ::Node;
                OCS1[xx]->setWeight((double)1.1);
                int index = _topo->addNode(OCS1[xx]);//Returns the index of the new graph node* (see getNode(int))

                int h,k;
                for(int q=0;q<(int)_topo->getNumNodes();q++){
                    if(_topo->getNode(q)->getModule()->getModuleType() == target->getModule()->getModuleType() && j->first == (int)_topo->getNode(q)->getModule()->getParentModule()->par("rackid")){
                        h = q;
                        break;
                    }
                }
                for(int q=0;q<(int)_topo->getNumNodes();q++){
                    if(_topo->getNode(q)->getModule()->getModuleType() == target->getModule()->getModuleType() && j->second == (int)_topo->getNode(q)->getModule()->getParentModule()->par("rackid")){
                        k = q;
                        break;
                    }
                }
                EV<<"OCS1 find corresponding node id: h = "<<h<<", k = "<<k<<endl;

                link1a[xx] = new cTopology::Link;
                link1b[xx] = new cTopology::Link;
//
                link1a[xx]->setWeight(1);
                link1b[xx]->setWeight(1);
//
                _topo->addLink(link1a[xx], _topo->getNode(h), _topo->getNode(index));//不知道是不是单向连接？
                _topo->addLink(link1b[xx], _topo->getNode(index), _topo->getNode(k));
                xx++;

            }
            int afterNodeNum = (int)_topo->getNumNodes();
            EV << "After CtrltoRoutingTableMod msg, cTopology found " << afterNodeNum << " nodes"<<endl;

            //calculate thisnode(target)'s routing table(destAddress,gateIndex)
            for (int i = 0; i < afterNodeNum; i++) {//destAdd
                 if (_topo->getNode(i) == target)
                      continue;  // skip ourselves
                 if (_topo->getNode(i)->getWeight() == (double)1 || _topo->getNode(i)->getWeight() == (double)1.1)//destAdd is OCS(which is impossible)
                      continue;
                 _topo->calculateWeightedSingleShortestPathsTo(_topo->getNode(i));

                 if (target->getNumPaths() == 0)
                      continue;  // not connected

 //                cGate *parentModuleGate = target->getPath(0)->getLocalGate();//is EPS Gate (= Routing Gate index)
//                 cGate *nextGate = parentModuleGate->getNextGate()->getNextGate();
//                 cModule * nextModule = nextGate->getOwnerModule();
//                 cTopology::Node *nextNode = _topo->getNodeFor(nextModule);

                 cTopology::Node *nextNode = target->getPath(0)->getRemoteNode();
                 //int nextNodeType = (int)target->getPath(0)->getLocalGate()->getNextGate()->getOwnerModule()->par("type");
                 int gateIndex;
                 //if (nextNodeType == 33 || nextNodeType == 44){
                     //gateIndex = parentModuleGate->getIndex();
                 //}
                 if (nextNode->getWeight() == (double)1){//next hop is ocs0
                     gateIndex = Nodenum+neighbor;
                 }
                 else if (nextNode->getWeight() == (double)1.1){//next hop is ocs1
                     gateIndex = Nodenum+neighbor+1;
                 }
                 else{
                     gateIndex = target->getPath(0)->getLocalGate()->getIndex();//->getBaseId();
                 }

                 int address;

                 if(_topo->getNode(i)->getModule()->getModuleType() != target->getModule()->getModuleType()){//->par("type") == 33){//if it's Node//不知道有没有问题！！
                     address= 10*(int)_topo->getNode(i)->getModule()->getParentModule()->par("rackid")+(int)_topo->getNode(i)->getModule()->par("id");
                 }
                 else if(_topo->getNode(i)->getModule()->getModuleType() == target->getModule()->getModuleType()){//->par("type") == 44){  //is EPS
                     address = 10000+(int)_topo->getNode(i)->getModule()->getParentModule()->par("rackid");
                 }
                 else{
                     EV<<"Node Type Error !" << endl;
                 }
                 rtable[address] = gateIndex;
            }
//            for (int i = 0; i < afterNodeNum; i++) {//destAdd
//                 if (_topo->getNode(i) == target)
//                      continue;  // skip ourselves
//                 _topo->calculateUnweightedSingleShortestPathsTo(_topo->getNode(i));
//
//                 if (target->getNumPaths() == 0)
//                      continue;  // not connected
//
//                 //cGate *parentModuleGate = target->getPath(0)->getLocalGate();//is EPS Gate (= Routing Gate index)
//                 int gateIndex;
//                 int gindex = target->getPath(0)->getLocalGate()->getIndex();
//
//                 if (gindex<(Nodenum+neighbor) && gindex>0){//forward to local Node or neighbor Rack
//                     gateIndex = gindex;
//                 }
//                 else {
////                     for (int qq = 0;qq < (int)ocs0map.size();qq++){//forward to OCS0
////                         if (gindex == ocs0index[qq]){
////                              gateIndex = Nodenum+neighbor;
////                         }
////                     }
////                     for (int pp = 0;pp < (int)ocs1map.size();pp++){//forward to OCS1
////                         if (gindex == ocs1index[pp]){
////                              gateIndex = Nodenum+neighbor+1;
////                         }
////                     }
//                     gateIndex
//                 }
//
//                 int address;
//
////                 cModuleType * mType = _topo->getNode(i)->getModule()->getModuleType();
////                 cModuleType * tType = target->getModule()->getModuleType();
////                 EV<<"mType:　"<<mType<<", tType: "<<tType<<endl;
//
//
//                 //else if(_topo->getNode(i)->getModule()->getModuleType() == target->getModule()->getModuleType()){//is EPS
//                 if((int)_topo->getNode(i)->getModule()->par("type") == 44){//is EPS
//                     address = 10000+(int)_topo->getNode(i)->getModule()->getParentModule()->par("rackid");
//                 }
//                 else if((int)_topo->getNode(i)->getModule()->par("type") == 33){//is Node
//                 //else if(_topo->getNode(i)->getModule()->getModuleType() != target->getModule()->getModuleType()){//if it's Node  不知道有没有问题！！
//                     address= 10*(int)_topo->getNode(i)->getModule()->getParentModule()->par("rackid")+(int)_topo->getNode(i)->getModule()->par("id");
//                 }
//                 else if(_topo->getNode(i)->getWeight()==(double)5 || _topo->getNode(i)->getWeight()==(double)6){
//                     EV<<"Node Type Error !" << endl;
//                 }
//                 rtable[address] = gateIndex;
//            }


            delete _topo;

    }
}


