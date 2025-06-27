import React from 'react'
import { Skeleton } from './ui/skeleton'

const Loading = () => {
  return (
    <main className="w-1/2 h-full flex flex-col justify-center items-start">
        <Skeleton className='h-5 w-2/3' />
        <Skeleton className='mt-5 h-10 w-full' />

        <div className='w-full flex gap-x-10 mt-20'>
            <Skeleton className='w-1/3 h-5' />
            <Skeleton className='w-2/3 h-5' />
        </div>

        <div className='w-full flex gap-x-10 mt-6'>
            <Skeleton className='w-1/3 h-5' />
            <Skeleton className='w-2/3 h-5' />
        </div>

        <Skeleton className='w-2/3 h-8 mt-10' />
        <Skeleton className='w-2/3 h-[130px] mt-4' />
    </main>
  )
}

export default Loading