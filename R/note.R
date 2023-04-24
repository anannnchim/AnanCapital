cmd <- cron_rscript("/Users/nanthawat/Desktop/R/Automated.R")
cron_add(command = cmd, frequency = 'minutely',
         at = "15:35" ,days_of_week = c(1:5), id = 'AAAA', description = "testing linux scheduler")



"/Library/Frameworks/R.framework/Resources/bin/Rscript '/Users/nanthawat/Desktop/R/Automated.R'  >> '/Users/nanthawat/Desktop/R/Automated.log' 2>&1"



# Example


cmd <- cron_rscript("/home/rstudio/my_folder/my_script.R")

[1] "/usr/lib/R/bin/Rscript '/home/rstudio/my_folder/my_script.R'  >> '/home/rstudio/my_folder/my_script.log' 2>&1"

cmd <- "cd '/home/rstudio/my_folder' && /usr/lib/R/bin/Rscript './my_script.R'  >> './my_script.log' 2>&1"
cron_add(command = cmd, frequency = 'daily', at = '18:00', id = 'test')



cmd <- "cd '/Users/nanthawat/Desktop/R/' && /Library/Frameworks/R.framework/Resources/bin/Rscript './Automated.R'  >> './Automated.log' 2>&1"
cron_add(command = cmd, frequency = 'daily', at = '18:00', id = 'test')



cmd <- cron_rscript("/Users/nanthawat/Desktop/R/Automated.R")
cron_add(command = cmd, frequency = 'minutely',
         at = "15:35" ,days_of_week = c(1:5), id = 'Stockscreener', description = "send df to googlesheet ")