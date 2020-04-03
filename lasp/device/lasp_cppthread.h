#pragma once
#ifndef LASP_CPPTHREAD_H
#define LASP_CPPTHREAD_H
#include <chrono>
#include <thread>
#include <assert.h>
// This is a small wrapper around the std library thread.

template <class T, typename F>
class CPPThread {
    std::thread _thread;
    public:
        CPPThread(F threadfcn, T data) :
            _thread(threadfcn, data) { }

        void join() {
            assert(_thread.joinable());
            _thread.join();
        }
        /* ~CPPThread() { */

        /* } */

};

void CPPsleep(unsigned int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
#endif // LASP_CPPTHREAD_H
