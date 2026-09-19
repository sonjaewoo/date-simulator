#ifndef SYNC_OBJECT_H
#define SYNC_OBJECT_H

#include <systemc.h>
#include <unordered_map>

class SyncObject
{
public:
    void signal(unsigned int sync_id)
    {
        sync[sync_id] = true;
        if (events.count(sync_id)) {
            events[sync_id]->notify();
        }
    }

    bool check_signal(unsigned int sync_id)
    {
        return sync[sync_id];
    }

    sc_event* get_event(unsigned int sync_id)
    {
        if (!events.count(sync_id)) {
            events[sync_id] = new sc_event;
        }
        return events[sync_id];
    }

    void free(unsigned int sync_id)
    {
        sync[sync_id] = false;
    }

private:
    std::unordered_map<unsigned int, bool> sync;
    std::unordered_map<unsigned int, sc_event*> events;
};

#endif
