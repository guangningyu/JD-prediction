/*
* @Author: han.jiang
* @Date:   2017-04-28 21:38:44
* @Last Modified by:   han.jiang
* @Last Modified time: 2017-05-09 11:48:15
*/

SELECT COUNT() FROM JDATA_USER_TBL;
-- 105321
SELECT COUNT() FROM JDATA_PRODUCT_TBL;
-- 24187
SELECT COUNT() FROM JDATA_COMMENT_TBL;
-- 558552
SELECT COUNT() FROM JDATA_ACTION_TBL;
-- 50601736



-- master table for model building
drop table if exists MASTER_0201_0408;
create table MASTER_0201_0408 as
select
A.*
,B.*
,C.*
from LABEL_0409_0413 as A
left join
SKU_MASTER_0201_0408 AS B
ON A.SKU_ID = B.SKU_ID
left join USER_ACTION_0201_0408 as C
ON A.SKU_ID = C.SKU_ID
left join
;

select count() from SKU_MASTER_0201_0410;
-- 27907


-- master table for model scoring
drop table if exists MASTER_0201_0415;
create table MASTER_0201_0415 as

