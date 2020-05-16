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

//#include "Queue.h"
#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include "..\Messages\Packet_m.h"

#include "Statistic.h"
using namespace omnetpp;

class ocs2rCheckBox : public cSimpleModule
{
  private:
    long Capacity;

    cQueue queue;
    cMessage *endTransmissionEvent;
    bool isBusy;
    int dropnum;

  public:
    ocs2rCheckBox();
    virtual ~ocs2rCheckBox();

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
    virtual void startTransmitting(cMessage *msg);
    virtual void finish();

    cStdDev dropsum;
};

Define_Module(ocs2rCheckBox);

ocs2rCheckBox::ocs2rCheckBox() {
    endTransmissionEvent = nullptr;
}

ocs2rCheckBox::~ocs2rCheckBox() {
    cancelAndDelete(endTransmissionEvent);
}

void ocs2rCheckBox::initialize()
{
    queue.setName("queue");
    endTransmissionEvent = new cMessage("endTxEvent");
    dropnum = 0;
    dropsum.setName("dropsum");

//    if (par("useCutThroughSwitching"))// use cut-through switching instead of store-and-forward
//        gate("line$i")->setDeliverOnReceptionStart(true);

    Capacity = 0.1; //par("frameCapacity");
    isBusy = false;
}

void ocs2rCheckBox::handleMessage(cMessage *msg)
{
    if (msg == endTransmissionEvent) {
        // Transmission finished, we can start next one.
        EV << "Transmission finished.\n";
        isBusy = false;
        if (queue.isEmpty()) {
            //emit(busySignal, false);
        }
        else {
            msg = (cMessage *)queue.pop();
            startTransmitting(msg);
        }
    }
//    else if (msg->arrivedOn("line$i")) {//
//        // pass up
//        //emit(rxBytesSignal, (long)check_and_cast<cPacket *>(msg)->getByteLength());
//        send(msg, "out");
//    }
    else {  // arrived on gate "in"
        if(msg->arrivedOn("in")) {
        if(dynamic_cast <Packet *> (msg) != NULL){
            Packet *pk = check_and_cast<Packet *>(msg);
            if (endTransmissionEvent->isScheduled()) {
                // We are currently busy, so just queue up the packet.
                if (Capacity && queue.getLength() >= Capacity) {
                    EV << "Received " << pk << " but transmitter busy and queue full: discarding\n";
                    //dropnum++;
                    //dropsum.collect(dropnum);
                    //Statistic::instance()->setLost(simTime(), pk->getSrcAddr(), pk->getDestAddr());
                    delete pk;
                }
                else {
                    EV << "Received " << pk << " but transmitter busy: queueing up\n";
                    pk->setTimestamp();
                    queue.insert(pk);
                }
            }
            else {
                // We are idle, so we can start transmitting right away.
                EV << "Received " << pk << "and it's ByteLength is : "<< pk->getByteLength()<<endl;
                startTransmitting(pk);
            }
        }
        }
    }
}

void ocs2rCheckBox::startTransmitting(cMessage *msg)
{
    if(dynamic_cast <Packet *> (msg) != NULL){
         Packet *pk = check_and_cast<Packet *>(msg);

         EV << "Starting transmission of " << pk << endl;
         isBusy = true;
         int64_t numBytes = check_and_cast<cPacket *>(msg)->getByteLength();
         send(pk, "line");//"line$o"

         // Schedule an event for the time when last bit will leave the gate.
         simtime_t endTransmission = gate("line")->getTransmissionChannel()->getTransmissionFinishTime();
         scheduleAt(endTransmission, endTransmissionEvent);
    }
}

void ocs2rCheckBox::finish(){
    //dropsum.record();
}
