import React from 'react';
import lodash from 'lodash';
import { helper } from './helpers';
import { api } from './api';
export function processData(inputData) {
  const temporaryResult = inputData.map((item) => {
    return item.value * 2;
  });
  const filteredData = temporaryResult.filter((x) => x > 10);
  const totalCount = filteredData.length;
  return { data: filteredData, count: totalCount };
}
