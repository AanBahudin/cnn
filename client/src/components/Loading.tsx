import React from 'react'
import { Skeleton } from './ui/skeleton'

const Loading = () => {
  return (
    <main className="w-1\3/5 h-full flex flex-col justify-center items-start">
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

        <section className='w-full flex gap-x-6 items-center justify-start mt-4'>
          {Array.from({length: 5}).map((_, index: number) => {
            return (
              <Skeleton key={index} className='w-[100px] h-[100px] rounded' />
            )
          })}
        </section>
    </main>
  )
}

export default Loading