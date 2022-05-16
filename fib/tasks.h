#pragma once

#include <memory>
#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

struct task
{
    virtual ~task() {}
    virtual void run() = 0;
};

class control_state
{
public:
    virtual ~control_state() {}
    virtual void at_exit() = 0;
};

class task_queue
{
    std::mutex mtx;
    std::condition_variable m_cvEmpty, m_cvNotEmpty;

    //std::queue<task*> m_q; //1
    std::queue<std::unique_ptr<task>> m_q; //2
public:
    inline void push_task(std::unique_ptr<task>&& tsk)
    {
        std::scoped_lock lck{mtx};
        //m_q.push(tsk.release()); //1
        m_q.push(std::move(tsk)); //2
        m_cvNotEmpty.notify_one();
    }
    inline std::unique_ptr<task> pop_task() //Blocking
    {
        std::unique_lock lck{mtx};
        while (m_q.empty())
            m_cvNotEmpty.wait(lck);
        //auto task = std::unique_ptr<task>(m_q.front()); // 1
        auto task = std::move(m_q.front()); //2
        m_q.pop();
        if (m_q.empty())
            m_cvEmpty.notify_all();
        return task;
    }

    inline void wait_until_empty()
    {
        std::unique_lock lck{mtx};
        while (!m_q.empty())
            m_cvEmpty.wait(lck);
    }
};

class thread_pool
{
    task_queue* m_q;
    std::atomic<bool> m_fStop;
    std::vector<std::thread> m_workers;
    inline void thread_proc()
    {
        while (!m_fStop)
        {
            auto task_ptr = m_q->pop_task(); //Cf. std::optional
            if (!!task_ptr)
                task_ptr->run();
        }
    }
public:
    inline thread_pool(unsigned T, task_queue* q):m_q(q), m_fStop(std::atomic<bool>(false))
    {
        for (unsigned t = 0; t < T; ++t)
            //m_workers.emplace_back(&thread_pool::thread_proc, this);
            m_workers.emplace_back([this]() {this->thread_proc();});
    }
    inline ~thread_pool()
    {
        m_fStop.store(true, std::memory_order_release);

        for (unsigned t = 0; t < m_workers.size(); ++t)
            m_q->push_task(nullptr);
        for (auto& thr:m_workers)
            thr.join();
    }
};

class load_balancer
{
    task_queue m_q;
    thread_pool m_pool;

    load_balancer():m_pool(std::thread::hardware_concurrency(), &m_q)
    {
    }
    load_balancer(load_balancer&&) = delete;
public:
    inline void add_task(std::unique_ptr<task>&& tsk)
    {
        m_q.push_task(std::move(tsk));
    }

    //Singleton
    inline static load_balancer& get()
    {
        static load_balancer lb{};
        return lb;
    }
};
