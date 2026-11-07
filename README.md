Student: Andrei Bors
Course: DTA400
University West Sweden, 2026

In the end, this simulation has been pretty much done by hand, overextended to a point where I absolutely realized that this has been way too much for the scope of the course.
While it has been a very interesting and rich learning experience, will have to scale down for the next opportunity.
AI (Main Claude) is seen and has been used in the project as a tool: brainstorming, case studying, research, code&bug_handling help, and main help with understanding the topic.
    -- The AI has NOT been brainlessly used to copy-paste code, as an engineering student -- i consider that shooting myself in the foot, I dispite vibe-coding.
    -- The initial and main reason for using AI tools for the project is my lack of knowledge over the area. Even after the project, I do not 100% understand how pytorch, agent training or other new tools work
      -- but I have the overall mastership over my own project.

-- Must run with requirements installed in the env.
-- Folders with the runs not included, must run enough of the simulation on your own if you actually want to see the statistics and contrasts in conditions.
--- Tipping to run it on a more powerfull CPU unit, the zombie experiment has a lot of steps in order to try to get actual learning from it.
--- If not interested in waiting that long for them, or no interest in stats at all, watching or saving a gif of a run is also possible
          -- specifically made for viewing, helps a lot with actually understanding the simulation

1. View/Test out:
   python watch.py --exp coin --cond proxy --seed 1
   python watch.py --exp coin --cond true --seed 1
   python watch.py --exp zombie --cond C --seed 1
   (view files for other conditions, such as extras or saving under .gif)
2. Run sim + create stats: (add the extra flag for extra personally studied edges to define behaviour)
   python run_all.py --exp coin --extra
   python run_all.py --exp zombie --extra
     -- the zombie env will likely take a very.long.time.
