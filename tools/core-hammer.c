/* core-hammer: drive ONE physical core's integer execution path, and say so in a log that survives the box.

   Why this exists: this box has died 18 times in two days, every death recorded by the firmware in ACPI BERT as
   an uncorrectable machine check whose IPID resolves to SMCA_EX -- a core's execution unit -- and the CPU number
   in every record is 17 or 113, which sysfs shows are the two SMT threads of physical core 65
   ([notes/incident-2026-11-03/README.md](../notes/incident-2026-11-03/README.md)). The test that turns "one core
   is blamed 18 times" into "one core is guilty" is to ask that core, and only that core, for integer work while
   the GPUs idle: if the same machine check comes back, the platform question collapses onto that core (or its
   rail); if a fault instead lands on a DIFFERENT core, the platform is the suspect; if nothing reproduces, the
   hammer is not the discriminator and the load that kills the box is something else.

   Deliberately integer-only: the EX bank covers fixed-point execution, and each SMCA bank names its own block,
   so mixing in heavy FP (FP bank) or memory pressure (UMC/LLC banks) would blur which unit is being asked.

   The log is flushed and fsynced every second on purpose. If the test "succeeds", the box resets underneath it,
   and the only thing that will show how long it ran, on which CPUs, and how far each worker got is the last line
   of a file that was fsynced before the death.

   usage: core-hammer [--seconds N] [--cpus 17,113] [--log PATH] [--stop FILE]
   pin:   the process pins one thread per listed logical CPU; siblings of the same core is the point.
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <stdatomic.h>

static volatile sig_atomic_t stop_now;
static atomic_long total_iters;
static atomic_long total_check;

static void on_sig(int s) { (void)s; stop_now = 1; }

/* Eight independent integer chains: multiply, add, xor-shift, compare-select, rotate. No FP, no allocation, no
   syscalls in the loop. The chains are independent so the scheduler and retire logic stay busy rather than a
   single dependent multiply chain sitting on latency -- EX faults are reported by the execution/scheduler logic,
   so occupancy is what is wanted, not latency. */
static void *hammer(void *arg)
{
    long cpu = (long)arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        fprintf(stderr, "core-hammer: pinning to CPU %ld failed (continuing unpinned)\n", cpu);

    unsigned long a = 0x243f6a8885a308d3UL + cpu, b = 0x13198a2e03707344UL, c = 0xa4093822299f31d0UL;
    unsigned long d = 0x082efa98ec1e6a14UL, e = 0x452821e638d01377UL, f = 0xbe5466cf34e90c6cUL;
    unsigned long g = 0xc0ac29b7c97c50ddUL, h = 0x3f84d5b5b5470917UL;
    unsigned long iters = 0, good = 0;

    while (!stop_now) {
        for (int k = 0; k < 20000; ++k) {
            a = a * 6364136223846793005UL + 1442695040888963407UL;
            b ^= (b << 13) ^ (b >> 7);
            c += (a ^ b) + 0x9e3779b97f4a7c15UL;
            d = (d << 21) | (d >> 43);
            e = e * 0x100000001b3UL ^ c;
            f += (d ^ e) | 1;
            g = (g >> 17) | (g << 47);
            h ^= (a + g) & 0xffffffffUL;
            if ((a ^ c ^ e ^ g) > (b ^ d ^ f ^ h))
                good++;
            iters++;
        }
        atomic_fetch_add(&total_iters, (long)iters);
        iters = 0;
        if (stop_now)
            break;
        /* A checksum the log can carry: proves the loop really ran arithmetic, not just spun. */
        atomic_fetch_add(&total_check, (long)(a ^ c ^ e ^ g ^ good));
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int seconds = 1800;
    const char *logpath = NULL;
    const char *stopfile = NULL;
    const char *cpus = "17,113";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpus") && i + 1 < argc) cpus = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) logpath = argv[++i];
        else if (!strcmp(argv[i], "--stop") && i + 1 < argc) stopfile = argv[++i];
        else { fprintf(stderr, "usage: core-hammer [--seconds N] [--cpus 17,113] [--log PATH] [--stop FILE]\n");
               return 2; }
    }

    long list[64];
    int n = 0;
    for (char *tok = strtok(strdup(cpus), ","); tok && n < 64; tok = strtok(NULL, ","))
        list[n++] = atol(tok);
    if (!n) { fprintf(stderr, "core-hammer: no CPUs given\n"); return 2; }

    /* Pin the whole process to the sibling pair, not only the workers: "this test touches nothing but core
       65" should be true of every thread the program has, so the experiment cannot perturb the cores the box
       is actually serving on. */
    {
        cpu_set_t all;
        CPU_ZERO(&all);
        for (int i = 0; i < n; ++i)
            CPU_SET(list[i], &all);
        if (sched_setaffinity(0, sizeof(all), &all) != 0)
            fprintf(stderr, "core-hammer: process affinity to %s failed\n", cpus);
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    FILE *log = logpath ? fopen(logpath, "a") : stdout;
    if (!log) { fprintf(stderr, "core-hammer: cannot open --log %s\n", logpath); return 2; }
    fprintf(stderr, "core-hammer: %d worker(s) on CPU(s) %s for %d s\n", n, cpus, seconds);

    pthread_t th[64];
    for (int i = 0; i < n; ++i)
        pthread_create(&th[i], NULL, hammer, (void *)list[i]);

    time_t end = time(NULL) + seconds;
    while (time(NULL) < end && !stop_now) {
        struct timespec one = {1, 0};
        nanosleep(&one, NULL);
        if (stopfile && access(stopfile, F_OK) == 0) { stop_now = 1; }
        if (!logpath) continue;
        /* Heartbeat, flushed and fsynced: after a reset this line is the whole record of the run. */
        fprintf(log, "%ld alive pid=%d cpus=%s seconds_left=%ld iters=%ld check=%lx\n",
                (long)time(NULL), (int)getpid(), cpus, (long)(end - time(NULL)),
                atomic_load(&total_iters), (unsigned long)atomic_load(&total_check));
        fflush(log);
        fsync(fileno(log));
    }
    stop_now = 1;
    for (int i = 0; i < n; ++i)
        pthread_join(th[i], NULL);
    fprintf(stderr, "core-hammer: done, %ld iterations total\n", atomic_load(&total_iters));
    if (logpath) fclose(log);
    return 0;
}
